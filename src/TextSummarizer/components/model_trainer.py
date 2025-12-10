import os
from src.TextSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments
import torch
from src.TextSummarizer.entity import ModelTrainerConfig
from datasets import DatasetDict

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ModelTrainer():
    def __init__(self, config: ModelTrainerConfig):
        self.config=config

    def train(self):
        device="cuda" if torch.cuda.is_available() else "cpu"
        tokenizer=AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model2 = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator=DataCollatorForSeq2Seq(tokenizer , model=model2)

        #loading data
        ds=load_from_disk(self.config.data_path)
        ds_train = ds["train"].select(range(2000))         # reduce size
        ds_valid = ds["validation"].select(range(300))

        trainer_args = TrainingArguments(
                output_dir=self.config.root_dir,

                
                num_train_epochs=0.3,                # ~1/3rd of an epoch
                max_steps=200,                       # Hard stop at 200 steps (≈ 10–15 min)

                
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,

                
                warmup_steps=10,                     # drastically reduced
                learning_rate=5e-5,                  # faster learning

                
                logging_steps=50,
                save_strategy="no",
                report_to="none",

                
                fp16=True,                           # faster on NVIDIA GPUs

                
                push_to_hub=False,
                dataloader_num_workers=0
            )

        
        trainer = Trainer(model=model2, args=trainer_args,
                    tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                    train_dataset=ds_train,
                    eval_dataset=ds_valid)
        trainer.train()

        model2.save_pretrained(os.path.join(self.config.root_dir, "pegasus_samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "Tokenizer")) 
