import logging
from datetime import datetime
import logging.config
import os

LOG_FILE_NAME=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
LOG_FOLDER="logs"

os.makedirs(LOG_FOLDER,exist_ok=True)
log_file=os.path.join(LOG_FOLDER,LOG_FILE_NAME)

logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

logger=logging.getLogger("Logger")
