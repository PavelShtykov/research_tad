# %load_ext autoreload
# %autoreload 2

import torch
from transformers import LlamaConfig, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from aux.layer0_llama import LlamaForCausalLM
from datasets import load_dataset 
from torch.utils.data import DataLoader
from datasets import Dataset

import torch
import litgpt
# import litdata as ld
# from litgpt.pretrain import initialize_weights
import lightning as L
import os
from aux.utils import provide_pajama

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# tokenizer_name = "meta-llama/Llama-3.2-1B"
tokenizer_name = "TinyLlama/TinyLlama_v1.1"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# model = AutoModelForCausalLM.from_pretrained(tokenizer_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_len_single_sentence=4)
# tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.model_max_length = 2048

config = LlamaConfig(
    vocab_size=tokenizer.vocab.__len__(),  # Use the same vocabulary size as the original model
    hidden_size=512,
    intermediate_size=1024*4,
    num_hidden_layers=2,#12,
    num_attention_heads=8,
    num_key_value_heads=8,
    head_dim=64,
    max_position_embeddings=tokenizer.model_max_length,  # Same as Llama-3
    rms_norm_eps=1e-5,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    # Additional Llama-3 specific parameters
    rope_theta=250000.0,
    attention_bias=False,
    tie_word_embeddings=True,
    model_type="llama",
    _attn_implementation='eager'
    # _attn_implementation_autoset=True,
)

model = LlamaForCausalLM(config)

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    print(f"Actual size of model: {params / 1024 ** 2:.0f}M")

count_trainable_parameters(model)

# model
# train_dataloader = provide_pajama(
#     train_batch=2, 
#     val_batch=None,
#     pad_token_id=tokenizer.pad_token_id, 
#     max_length=4, 
# )

inputs = {
    'input_ids': torch.tensor([[1, 2, tokenizer.pad_token_id, tokenizer.pad_token_id], [1, 3, 4, tokenizer.pad_token_id]]),
    'attention_mask': torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
}

# inputs = next(iter(train_dataloader))
print(inputs)
model.train()
outputs = model(**inputs)
print(tokenizer.batch_decode(outputs))
