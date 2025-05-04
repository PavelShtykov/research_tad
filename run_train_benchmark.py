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
import statistics

from aux.utils import provide_pajama, get_git_info
from aux.arch_mod import LitLLM, LlamaForCausalLMFirstLayer, LlamaForCausalLM

os.environ['TORCH_LOGS'] = "+dynamo"
os.environ['TORCHDYNAMO_VERBOSE'] = '1'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Benchmark(L.Callback):
    """A callback that measures the median execution time between the start and end of a batch."""
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.times = []

    def median_time(self):
        return statistics.median(self.times)

    def on_train_batch_start(self, trainer, *args, **kwargs):
        self.start.record()

    def on_train_batch_end(self, trainer, *args, **kwargs):
        # Exclude the first iteration to let the model warm up
        if trainer.global_step > 1:
            self.end.record()
            torch.cuda.synchronize()
            self.times.append(self.start.elapsed_time(self.end) / 1000)


# ============================================================================================================================
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
MICRO_BATCH = 8
ACCUMULATED_BATCHES = 10
TARGET_GLOBAL_BATCH = 1024
VAL_BATCH = 8
TARGET_TRAIN_TOKENS = 1_000_000_000
MAX_LENGTH = 1024

GLOBAL_BATCH = ACCUMULATED_BATCHES * MICRO_BATCH # ~1k
# assert abs(TARGET_GLOBAL_BATCH - GLOBAL_BATCH) / TARGET_GLOBAL_BATCH < 0.07
ITERS = TARGET_TRAIN_TOKENS // 700
MAX_STEPS = 10


logger.info(f'TARGET_TRAIN_TOKENS: {TARGET_TRAIN_TOKENS}; GLOBAL_BATCH: {GLOBAL_BATCH}; MICRO_BATCH: {MICRO_BATCH}; ACCUMULATED_BATCHES: {ACCUMULATED_BATCHES}; MAX_STEPS: {MAX_STEPS}; ITERS: {ITERS}; VAL_BATCH: {VAL_BATCH}; MAX_LENGTH: {MAX_LENGTH}')
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
    # _attn_implementation='eager',
    _attn_implementation_my = 'layer0_attention_eager'
    # _attn_implementation_autoset=True,
)

# lit_llm = LitLLM(config, LlamaForCausalLMFirstLayer, use_compile=False, total_steps=MAX_STEPS)
# benchmark = Benchmark()

# trainer = L.Trainer(
#     # logger=wandb_logger,
#     callbacks=[benchmark],
#     # log_every_n_steps=4,
#     max_steps=5, 
#     limit_train_batches=ITERS // MICRO_BATCH,
#     accumulate_grad_batches=ACCUMULATED_BATCHES, 
#     val_check_interval=1000,
#     limit_val_batches=8,
#     # num_sanity_val_steps=4,

#     accelerator="gpu", devices=1, 
#     precision='bf16-mixed',
#     # fast_dev_run=True
# )

# trainer.fit(
#     model=lit_llm, 
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader
# )


# eager_time = benchmark.median_time()


# del lit_llm, trainer, benchmark

lit_llm = LitLLM(config, LlamaForCausalLMFirstLayer, use_compile=True, total_steps=MAX_STEPS)
benchmark = Benchmark()

trainer = L.Trainer(
    # logger=wandb_logger,
    callbacks=[benchmark],
    # log_every_n_steps=4,
    max_steps=5, 
    limit_train_batches=ITERS // MICRO_BATCH,
    accumulate_grad_batches=ACCUMULATED_BATCHES, 
    val_check_interval=10000,
    limit_val_batches=8,
    # num_sanity_val_steps=4,

    accelerator="gpu", devices=1, 
    precision='bf16-mixed',
    # fast_dev_run=True
)

trainer.fit(
    model=lit_llm, 
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)


compile_time = benchmark.median_time()


# speedup = eager_time / compile_time
print('================ Eager layer0 ==================')
# print(f"Eager median time: {eager_time:.4f} seconds")
print(f"Compile median time: {compile_time:.4f} seconds")
# print(f"Speedup: {speedup:.2f}x")


del lit_llm, trainer, benchmark


# config = LlamaConfig(
#     vocab_size=tokenizer.vocab.__len__(),  # 32k size
#     hidden_size=512,
#     intermediate_size=512*4,
#     num_hidden_layers=12,
#     num_attention_heads=16,
#     num_key_value_heads=16,
#     head_dim=32,
#     max_position_embeddings=tokenizer.model_max_length,  # 2048 tokens
#     rms_norm_eps=1e-5,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.pad_token_id,
#     # Additional Llama-3 specific parameters
#     rope_theta=50000.0,
#     attention_bias=False,
#     tie_word_embeddings=True,
#     model_type="llama",
#     torch_dtype='bfloat16',
#     # _attn_implementation='eager',
#     _attn_implementation_my = 'layer0_attention_flex'
#     # _attn_implementation_autoset=True,
# )

# lit_llm = LitLLM(config, LlamaForCausalLMFirstLayer, use_compile=False, total_steps=MAX_STEPS)
# benchmark = Benchmark()

# trainer = L.Trainer(
#     # logger=wandb_logger,
#     callbacks=[benchmark],
#     # log_every_n_steps=4,
#     max_steps=5, 
#     limit_train_batches=ITERS // MICRO_BATCH,
#     accumulate_grad_batches=ACCUMULATED_BATCHES, 
#     val_check_interval=1000,
#     limit_val_batches=8,
#     # num_sanity_val_steps=4,

#     accelerator="gpu", devices=1, 
#     precision='bf16-mixed',
#     # fast_dev_run=True
# )

# trainer.fit(
#     model=lit_llm, 
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader
# )


# eager_time = benchmark.median_time()


# del lit_llm, trainer, benchmark

# lit_llm = LitLLM(config, LlamaForCausalLMFirstLayer, use_compile=True, total_steps=MAX_STEPS)
# benchmark = Benchmark()

# trainer = L.Trainer(
#     # logger=wandb_logger,
#     callbacks=[benchmark],
#     # log_every_n_steps=4,
#     max_steps=5, 
#     limit_train_batches=ITERS // MICRO_BATCH,
#     accumulate_grad_batches=ACCUMULATED_BATCHES, 
#     val_check_interval=10000,
#     limit_val_batches=8,
#     # num_sanity_val_steps=4,

#     accelerator="gpu", devices=1, 
#     precision='bf16-mixed',
#     # fast_dev_run=True
# )

# trainer.fit(
#     model=lit_llm, 
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader
# )


# compile_time = benchmark.median_time()


# speedup = eager_time / compile_time
# print('================ Flex attn layer0 ==================')
# print(f"Eager median time: {eager_time:.4f} seconds")
# print(f"Compile median time: {compile_time:.4f} seconds")
# print(f"Speedup: {speedup:.2f}x")
