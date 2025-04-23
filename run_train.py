import torch
torch.set_float32_matmul_precision("high")

from transformers import LlamaConfig, AutoTokenizer

from torch.utils.data import DataLoader

import torch
import lightning as L
import os
import logging
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import datetime

from aux.utils import provide_pajama
from aux.arch_mod import Mods, provide_litllm_with_mod


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================================================================
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
MICRO_BATCH = 8
ACCUMULATED_BATCHES = 128
TARGET_GLOBAL_BATCH = 1024
VAL_BATCH = 8
TARGET_TRAIN_TOKENS = 1_000_000_000
MAX_LENGTH = 1024
MODS_LIST = [Mods.first_layer_key_value]

GLOBAL_BATCH = ACCUMULATED_BATCHES * MICRO_BATCH # ~1k
assert abs(TARGET_GLOBAL_BATCH - GLOBAL_BATCH) / TARGET_GLOBAL_BATCH < 0.07
ITERS = TARGET_TRAIN_TOKENS // 700
MAX_STEPS = ITERS // GLOBAL_BATCH


logger.info(f'TARGET_TRAIN_TOKENS: {TARGET_TRAIN_TOKENS}; GLOBAL_BATCH: {GLOBAL_BATCH}; MICRO_BATCH: {MICRO_BATCH}; ACCUMULATED_BATCHES: {ACCUMULATED_BATCHES}; MAX_STEPS: {MAX_STEPS}; ITERS: {ITERS}; VAL_BATCH: {VAL_BATCH}; MAX_LENGTH: {MAX_LENGTH}, MODS: {MODS_LIST}')
# ============================================================================================================================



tokenizer_name = "TinyLlama/TinyLlama_v1.1"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_len_single_sentence=MAX_LENGTH)
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
    hidden_size=512,
    intermediate_size=512*4,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=32,
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
    _attn_implementation='eager'
    # _attn_implementation_autoset=True,
)

lit_llm = provide_litllm_with_mod(
    llama_config=config, total_steps=MAX_STEPS,
    mods=MODS_LIST,
    max_lr=3e-4, warmup=0.05,
)

mods_str = '+'.join(
    "".join([p[0] for p in m.name.split('_')]) 
    for m in MODS_LIST
) 
wandb_logger = WandbLogger(
    project=f'base_microllama_64m',
    version=f'v_{mods_str}_{TARGET_TRAIN_TOKENS // 10**6}Mtok_{MAX_STEPS}steps_{GLOBAL_BATCH}gbs',
    # log_model=False,
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

    accelerator="gpu", devices=1, 
    precision='bf16-mixed',
    # fast_dev_run=True
)

trainer.fit(
    model=lit_llm, 
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)
