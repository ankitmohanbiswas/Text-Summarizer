from src.TextSummarizer.logging import logger
from src.TextSummarizer.pipeline.stage1_DATA_ING import DataIngestionTrainingPipeline


STAGE_NAME="Data Ingestion Stage"

try:
    logger.info(f"stage {STAGE_NAME} has started")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.inititate_data_ingestion()
    logger.info(f"{STAGE_NAME} has completed")

except Exception as e:
    logger.exception(e)
    raise e