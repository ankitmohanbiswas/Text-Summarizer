from src.TextSummarizer.constants import *
from src.TextSummarizer.utils.common import read_yaml, create_directories
from src.TextSummarizer.entity import DataIngestionConfig
from src.TextSummarizer.entity import DataTransformationConfig
from src.TextSummarizer.entity import ModelTrainerConfig
from src.TextSummarizer.entity import ModelEvaluationConfig

class ConfigurationManager:
    def __init__(self,
                config_path=CONFIG_FILE_PATH,
                params_path=PARAMS_FILE_PATH):
        self.config= read_yaml(config_path)
        self.params=read_yaml(params_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config= DataIngestionConfig(
            root_dir= Path(config.root_dir),
            source_url= config.source_url,
            local_data_file= Path(config.local_data_file),
            unzip_dir= Path(config.unzip_dir)
        )
        return data_ingestion_config

    def get_data_transformation_config(self)-> DataTransformationConfig:
        config=self.config.data_transformation

        data_transformation_config=DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )

        return data_transformation_config
        

    def get_model_trainer_config(self)->ModelTrainerConfig:
        config=self.config.model_trainer
        parmas=self.params.TrainingArguments

        model_trainer_config=ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=config.model_ckpt,
            num_train_epochs=parmas.num_train_epochs,
            per_device_train_batch_size=parmas.per_device_train_batch_size,
            per_device_eval_batch_size=parmas.per_device_eval_batch_size,
            gradient_accumulation_steps=parmas.gradient_accumulation_steps,
            warmup_steps=parmas.warmup_steps,
            learning_rate=parmas.learning_rate,
            logging_steps=parmas.logging_steps,
            fp16=parmas.fp16,
            save_strategy=parmas.save_strategy,
            report_to=parmas.report_to,
            max_steps=200,
            push_to_hub=False
                )
        return model_trainer_config
    
    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        config= self.config.model_evaluation
        model_evaluation_config= ModelEvaluationConfig(
            root_dir= config.root_dir,
            data_path= config.data_path,
            model_path= config.model_path,
            tokenizer_path= config.tokenizer_path,
            metric_file_name= config.metric_file_name
        )
        return model_evaluation_config