from src.TextSummarizer.logging import logger
from src.TextSummarizer.pipeline.stage1_DATA_ING import DataIngestionTrainingPipeline
from src.TextSummarizer.pipeline.stage2_data_transformation import DataTransformationPipeline

STAGE_NAME="Data Ingestion Stage"

try:
    logger.info(f"stage {STAGE_NAME} has started")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.inititate_data_ingestion()
    logger.info(f"{STAGE_NAME} has completed")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Data Transformation Stage"

try:
    logger.info(f"stage {STAGE_NAME} has started")
    data_transformation_pipeline=DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"{STAGE_NAME} has completed")

except Exception as e:
    logger.exception(e)
    raise e