import logging, os

logging.basicConfig(level=os.environ['LOG_LEVEL'], format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
