import os, sys
import logging
from datasets import load_dataset, Dataset, DatasetInfo, NamedSplit
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

tokenizer_name = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=2048)
tokenizer.pad_token = tokenizer.eos_token

def process_and_save_dataset(split, num_samples, buffer_size, num_shards, output_path_template):
    dataset = load_dataset('cerebras/SlimPajama-627B', split=split, streaming=True)
    dataset = (dataset
        .shuffle(seed=42, buffer_size=buffer_size)
        .map(
            lambda x: tokenizer(x['text']), 
            batched=True, batch_size=1000
        )
    )

    shard_size = num_samples // num_shards
    it = iter(dataset)
    
    for shard_index in range(num_shards):

        res = []
        skipped = 0

        for _ in tqdm(range(shard_size), desc=f"Processing shard {shard_index+1}/{num_shards} for {split}"):
            try:
                res.append(next(it))
            except StopIteration:
                skipped += 1

        logging.info(f'skipped in {split}: {skipped}')

        shard_dataset = Dataset.from_list(
            res, 
            info=DatasetInfo(
                dataset_name='SlimPajama-627B',
                description=f'Part of {split} dataset, processed with llama3 tokenizer',
                dataset_size=len(res),
                size_in_bytes=sys.getsizeof(res)
            ),
            split=NamedSplit(split)
        )
        
        shard_dataset.to_parquet(output_path_template.format(index=shard_index))
        del res, shard_dataset # Освобождаем память после записи каждого шарда


process_and_save_dataset(
    split='train', 
    num_samples=5_000_000, 
    buffer_size=50_000, 
    num_shards=320, 
    output_path_template="./slim_pajama_5m/train/{index:05d}.parquet"
)

process_and_save_dataset(
    split='validation', 
    num_samples=50_000, 
    buffer_size=10_000, 
    num_shards=16, 
    output_path_template="./slim_pajama_5m/validation/{index:05d}.parquet"
)
