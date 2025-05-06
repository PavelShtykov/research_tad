import torch
torch.set_float32_matmul_precision("high")

from transformers import LlamaConfig, AutoTokenizer

from torch.utils.data import DataLoader
import torch._dynamo

import torch
import lightning as L
import os
import logging
import argparse
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import datetime
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

from aux.utils import provide_pajama, get_git_info
from aux.arch_mod import LitLLM

os.environ['TORCH_LOGS'] = "+dynamo"
os.environ['TORCHDYNAMO_VERBOSE'] = '1'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================================================================
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
MICRO_BATCH = 20
ACCUMULATED_BATCHES = 12
TARGET_GLOBAL_BATCH = 256
VAL_BATCH = 8
TARGET_TRAIN_TOKENS = 2_000_000_000
MAX_LENGTH = 1024
LLAMA_CLASS = 'Base_Llama'
MODS_LIST = []

GLOBAL_BATCH = ACCUMULATED_BATCHES * MICRO_BATCH
assert abs(TARGET_GLOBAL_BATCH - GLOBAL_BATCH) / TARGET_GLOBAL_BATCH < 0.10
ITERS = TARGET_TRAIN_TOKENS // 700
MAX_STEPS = ITERS // GLOBAL_BATCH
_TOKENS_PER_STEP = GLOBAL_BATCH * 700


logger.info(f'{TARGET_TRAIN_TOKENS=}; {GLOBAL_BATCH=}; {MICRO_BATCH=}; {ACCUMULATED_BATCHES=}; {MAX_STEPS=}; {ITERS=}; {VAL_BATCH=}; {MAX_LENGTH=}; {_TOKENS_PER_STEP=}; {MODS_LIST=}; {LLAMA_CLASS=}')
# ============================================================================================================================
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train LLM model with optional debug mode')
parser.add_argument('--debug', action='store_true', help='Enable debug mode (disables wandb logging and callbacks)')
args = parser.parse_args()

# ============================================================================================================================



tokenizer = LlamaTokenizerFast.from_pretrained('./aux/tokenizer', max_len_single_sentence=MAX_LENGTH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_LENGTH

train_dataloader, val_dataloader = provide_pajama(
    train_batch=MICRO_BATCH, 
    val_batch=VAL_BATCH,
    pad_token_id=tokenizer.pad_token_id, 
    max_length=MAX_LENGTH, 
)


config = LlamaConfig(
    vocab_size=tokenizer.vocab.__len__(),  # 32k size
    hidden_size=768,
    intermediate_size=768*4,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=12,
    head_dim=64,
    max_position_embeddings=tokenizer.model_max_length,  # 2048 tokens
    rms_norm_eps=1e-5,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    # Additional Llama-3 specific parameters
    rope_theta=50000.0,
    attention_bias=False,
    tie_word_embeddings=True,
    model_type="llama",
    torch_dtype='bfloat16',
    _attn_implementation='sdpa',
    # _attn_implementation_my='layer0_attention_eager'
    # _attn_implementation_autoset=True,
)

lit_llm = LitLLM(
    config, LLAMA_CLASS, MODS_LIST,
    use_compile=True, 
    max_lr=1e-3, warmup=0.05, total_steps=MAX_STEPS
)

if args.debug:
    logger.info("Debug mode enabled: wandb_logger and callbacks will not be used")
    wandb_logger = None
    callbacks = None
else:
    mods_str = '+'.join(
        "".join([p[0] for p in m.name.split('_')]) 
        for m in MODS_LIST
    ) 
    llama_class_str = "".join([p[0] for p in LLAMA_CLASS.split('_')]) 
    wandb_logger = WandbLogger(
        project=f'base_microllama_170m',
        version=f'v_{llama_class_str}_{mods_str}_{TARGET_TRAIN_TOKENS // 10**6}Mtok_{MAX_STEPS}steps_{GLOBAL_BATCH}gbs',
        # log_model=False,
        config={
            'micro_batch': MICRO_BATCH,
            'accumulated_batches': ACCUMULATED_BATCHES,
            'global_batch': GLOBAL_BATCH,
            'target_tokens': TARGET_TRAIN_TOKENS,
            'max_steps': MAX_STEPS,
            'mods': [mod.name for mod in MODS_LIST],
            'git': get_git_info(),
            'llama_config': config.to_dict()
        }
    )
    wandb_logger.watch(lit_llm)

    callbacks = [
        ModelCheckpoint(
            # every_n_train_steps=10,
            filename='step={step}-val_loss={val/loss:.2f}',
            save_top_k=2,
            monitor='val/loss',
            save_on_train_epoch_end=True,
            train_time_interval=datetime.timedelta(minutes=10),
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='step')
    ]


# trainer = L.Trainer(
#     fast_dev_run=10,
#     accelerator="gpu", devices=[0, 1], 
#     precision='bf16-mixed',
#     strategy=DDPStrategy(static_graph=True,)
# )
# trainer.fit(
#     model=lit_llm, 
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader
# )


trainer = L.Trainer(
    logger=wandb_logger,
    callbacks=callbacks,
    log_every_n_steps=4,
    max_steps=MAX_STEPS, 
    limit_train_batches=ITERS // MICRO_BATCH,
    accumulate_grad_batches=ACCUMULATED_BATCHES, 
    val_check_interval=2*ACCUMULATED_BATCHES,
    limit_val_batches=8,
    num_sanity_val_steps=4,

    accelerator="gpu", devices=[0, 1], 
    precision='16-mixed',
    strategy='ddp'
    # fast_dev_run=True
)

trainer.fit(
    model=lit_llm, 
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)
