from aux.utils import wait_for_process_to_finish
wait_for_process_to_finish(2401936)


import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, IterableDataset
import datasets
datasets.config.STREAMING_READ_MAX_RETRIES = 30
torch.set_float32_matmul_precision("high")

from torch.utils.data import DataLoader

import torch
import lightning as L
import os
import logging
from litgpt.utils import chunked_cross_entropy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import datetime
from aux.arch_mod import mod_parallel_attention


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
MICRO_BATCH = 24
ACCUMULATED_BATCHES = 42
GLOBAL_BATCH = ACCUMULATED_BATCHES * MICRO_BATCH # ~1k
VAL_BATCH = 8
TARGET_TRAIN_TOKENS = 1_000_000_000
ITERS = TARGET_TRAIN_TOKENS // 700
MAX_STEPS = ITERS // GLOBAL_BATCH
MAX_LENGTH = 1024

logger.info(f'TARGET_TRAIN_TOKENS: {TARGET_TRAIN_TOKENS}; GLOBAL_BATCH: {GLOBAL_BATCH}; MICRO_BATCH: {MICRO_BATCH}; ACCUMULATED_BATCHES: {ACCUMULATED_BATCHES}; MAX_STEPS: {MAX_STEPS}; ITERS: {ITERS}; VAL_BATCH: {VAL_BATCH}; MAX_LENGTH: {MAX_LENGTH}')

tokenizer_name = "TinyLlama/TinyLlama_v1.1"

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_len_single_sentence=MAX_LENGTH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_LENGTH


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
    _attn_implementation='flash_attention_2'
    # _attn_implementation_autoset=True,
)


model = LlamaForCausalLM(config)
model = mod_parallel_attention(model)
# model.half()
# logger.info(model.dtype)


def collate_fn(batch, pad_idx, max_len=MAX_LENGTH):
    
    batch_max = min(max_len, max(map(lambda x: len(x['input_ids']), batch)))

    ret = dict()
    ret['input_ids'] = torch.tensor([
        x['input_ids'][:batch_max] + [pad_idx] * max(0, batch_max - len(x['input_ids']))
        for x in batch
    ])
    ret['attention_mask'] = torch.tensor([
        x['attention_mask'][:batch_max] + [0] * max(0, batch_max - len(x['attention_mask']))
        for x in batch
    ])
    
    return ret

data_files = {"train": "train/*.parquet", "validation": "validation/*.parquet"}
dataset: IterableDataset = load_dataset("slim_pajama_1m", data_files=data_files, streaming=True)
dataset = dataset.filter(
    lambda b: b['meta']['redpajama_set_name'] not in {'RedPajamaStackExchange', 'RedPajamaGithub', 'RedPajamaArXiv'},
    # batch_size=5000,
    # num_proc=6
)
dataset = dataset.remove_columns(['text', 'meta'])

train_dataloader = DataLoader(
    dataset['train'], batch_size=MICRO_BATCH, shuffle=False, 
    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, MAX_LENGTH), 
    num_workers=min(MICRO_BATCH, 6),
    # pin_memory=True
)

val_dataloader = DataLoader(
    dataset['validation'], batch_size=VAL_BATCH, shuffle=False, 
    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, MAX_LENGTH), 
    num_workers=min(VAL_BATCH, 6),
    # pin_memory=True
)


one_batch = next(iter(train_dataloader))
logger.info(f'Example one batch:\n {one_batch}')


class LitLLM(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._consumed_tokens = 0
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        predict = outputs['logits'][..., :-1, :]
        target = batch["input_ids"][..., 1:]
        loss = chunked_cross_entropy(predict, target, chunk_size=256, ignore_index=self.model.config.pad_token_id)

        self.log("train/loss", loss, prog_bar=True)
        # self.log('train/steps', batch_idx // ACCUMULATED_BATCHES, prog_bar=True)

        self._consumed_tokens += batch['attention_mask'].sum().item()
        self.log('train/consumed_tokens', self._consumed_tokens)
        return loss
    
    def validation_step(self, batch):
        outputs = self.model(**batch)
        predict = outputs['logits'][..., :-1, :]
        target = batch["input_ids"][..., 1:]
        loss = chunked_cross_entropy(predict, target, chunk_size=256, ignore_index=self.model.config.pad_token_id)

        self.log("val/loss", loss, prog_bar=True)


    def configure_optimizers(self):
        # warmup_steps = int(0.15 * MAX_STEPS)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=MAX_STEPS, pct_start=0.05, anneal_strategy='linear', )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

lit_llm = LitLLM(model=model)
wandb_logger = WandbLogger(
    project=f'base_microllama_64m',
    version=f'v_parattn_{TARGET_TRAIN_TOKENS // 10**6}Mtok_{MAX_STEPS}steps_{GLOBAL_BATCH}gbs',
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
    log_every_n_steps=4,
    callbacks=callbacks,
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
