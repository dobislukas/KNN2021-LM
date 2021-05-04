from typing import Optional

import pytorch_lightning as pl
import torch
import os
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from torch.utils.data.dataloader import DataLoader
from tokenizers import ByteLevelBPETokenizer


class WikiText2DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'data/wikitext-2', train_batch_size: int = 64, val_batch_size: int = 64,
                 dataloader_num_workers: int = 4, seq_length: int = 64, vocab_size=30000):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        self.tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

    def prepare_data(self, *args, **kwargs):
        dataset = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train+test+validation"
        )
        column_names = dataset.column_names

        def batch_iterator(batch_size=1000):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i: i + batch_size]["text"]

        if not os.path.exists("data/wiki-vocab.json") and not os.path.exists("data/wiki-merges.json"):
            print('here')
            self.tokenizer.train_from_iterator(batch_iterator(), vocab_size=self.vocab_size)
            self.tokenizer.save_model("data/", "wiki")
        else:
            self.tokenizer = ByteLevelBPETokenizer("data/wiki-vocab.json", "data/wiki-merges.json",
                                                   add_prefix_space=True)

        dataset = load_dataset(
            "wikitext", "wikitext-103-raw-v1"
        )

        def tokenize_function(examples):
            return {
                'input_ids':
                    list(map(lambda x: x.ids, self.tokenizer.encode_batch(examples['text'])))
            }

        self.tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=4
        )

    def setup(self, stage: Optional[str] = None):
        # datasets = load_dataset('text',
        #                         data_dir=self.data_dir,
        #                         data_files={'train': 'wiki.train.small.raw',
        #                                     'valid': 'wiki.valid.small.raw'})

        def group_text(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // self.seq_length) * self.seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + self.seq_length] for i in range(0, total_length, self.seq_length)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_dataset = self.tokenized_dataset.map(
            group_text,
            batched=True,
            num_proc=4
        )

        train_dataset = lm_dataset['train']
        eval_dataset = lm_dataset['validation']
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = lm_dataset['test']

    def collate_fn(self, features):
        batch = {}
        batch['inputs_ids'] = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        batch['labels'] = batch['inputs_ids']
        return batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.collate_fn,
            # num_workers=self.dataloader_num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            # num_workers=self.dataloader_num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            # num_workers=self.dataloader_num_workers
        )
