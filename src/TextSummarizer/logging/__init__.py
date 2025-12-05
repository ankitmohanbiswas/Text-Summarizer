import os
import sys
import logging
#-----------------------------------------
log_dir="logs"
logging_str= "[%(asctime)s: %(levelname)s: %(module)s: %(message)s:]"
log_file_path=os.path.join(log_dir,"continuous_logs.log")
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,

    handlers=[
        logging.FileHandler(log_file_path), #if an error happens the log msg will be saved in this path
        logging.StreamHandler(sys.stdout) # displayes all the logs in output terminal
    ]
)
logger=logging.getLogger("summarizerlogger")
