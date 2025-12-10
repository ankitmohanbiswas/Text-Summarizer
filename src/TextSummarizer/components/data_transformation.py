import os
import urllib.request as request
import zipfile
from src.TextSummarizer.logging import logger
from src.TextSummarizer.entity import DataIngestionConfig
from src.TextSummarizer.entity import DataTransformationConfig
from transformers import AutoTokenizer
from datasets import load_from_disk
from datasets import DatasetDict


class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config=config
        self.tokenizer=AutoTokenizer.from_pretrained(self.config.tokenizer_name)


    def cvrt_examples_to_features(self,ex_batch):
        
        input_encodings=self.tokenizer(ex_batch['dialogue'], max_length=511, truncation=True)
        with self.tokenizer.as_target_tokenizer():
         target_encodings=self.tokenizer(ex_batch['summary'], max_length=180, truncation=True)

        return {
            'input_ids':input_encodings['input_ids'],
            'attention_mask' :input_encodings['attention_mask'],
            'labels':target_encodings['input_ids']

            }

    def convert(self):
        ds=load_from_disk(self.config.data_path)
        small_train=ds['train'].select(range(2000))
        small_valid=ds['validation'].select(range(300))
        ds=DatasetDict({
           'train': small_train,
           'validation': small_valid
        })
        ds1=ds.map(self.cvrt_examples_to_features, batched=True)
        ds1.save_to_disk(os.path.join(self.config.root_dir, "transformed_dataset"))
