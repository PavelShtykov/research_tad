import os
import time
import sys
import logging
from typing import Callable, List, Tuple, Dict, Any, Optional
import torch
from datasets import load_dataset, IterableDataset
import git

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_git_info() -> Dict[str, str]:
    try:
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
        git_branch = repo.active_branch.name
        return {
            'git_hash': git_hash,
            'git_branch': git_branch
        }
    except Exception as e:
        logger.warning(f"Не удалось получить Git информацию: {e}")
        return {
            'git_hash': 'unknown',
            'git_branch': 'unknown'
        }


def wait_for_process_to_finish(pid: int):
    try:
        pid = int(pid)
    except ValueError:
        logging.error(f"Ошибка: PID должен быть числом, получено: {pid}")
        return
    
    logging.info(f"Ожидание завершения процесса с PID {pid}...")
    
    try:
        while True:
            try:
                os.kill(pid, 0)
                time.sleep(10)
            except OSError:
                time.sleep(10)
                logging.info(f"Процесс с PID {pid} завершился, скрипт разблокирован")
                break
    except KeyboardInterrupt:
        logging.error("\nОжидание процесса прервано пользователем")


def _build_dataset(
        path: str,
        train_batch: int, 
        pad_token_id: int,
        max_length: int = 1024,
        val_batch: Optional[int] = None,
        filter_fn: Optional[Callable] = None,
        data_files: Dict[str, str] = {"train": "train/*.parquet", "validation": "validation/*.parquet"}
    ):
    def collate_fn(batch, pad_idx, max_len):
        
        # batch_max = min(max_len, max(map(lambda x: len(x['input_ids']), batch)))
        batch_max = max_len
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
    
    dataset: IterableDataset = load_dataset(path, data_files=data_files, streaming=True)
    dataset = dataset.filter(filter_fn)
    dataset = dataset.remove_columns(['text', 'meta'])

    train_dataloader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=train_batch, shuffle=False, 
        collate_fn=lambda b: collate_fn(b, pad_token_id, max_length), 
        num_workers=min(train_batch, 6),
        # pin_memory=True
    )

    if val_batch:
        val_dataloader = torch.utils.data.DataLoader(
            dataset['validation'], batch_size=val_batch, shuffle=False, 
            collate_fn=lambda b: collate_fn(b, pad_token_id, max_length), 
            num_workers=min(val_batch, 6),
            # pin_memory=True
        )
    
    if val_batch:
        return train_dataloader, val_dataloader
    
    return train_dataloader


def provide_pajama(
        train_batch: int, 
        pad_token_id: int,
        max_length: int = 1024,
        val_batch: Optional[int] = None, 
    ):
    return _build_dataset(
        path='aux/slim_pajama_3m',
        train_batch=train_batch,
        pad_token_id=pad_token_id,
        max_length=max_length,
        val_batch=val_batch,
        filter_fn=lambda b: b['meta']['redpajama_set_name'] not in {'RedPajamaStackExchange', 'RedPajamaGithub', 'RedPajamaArXiv'},
    )
