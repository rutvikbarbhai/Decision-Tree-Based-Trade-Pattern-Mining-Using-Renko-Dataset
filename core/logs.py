import logging
import sys

# Logging Definitions
log_lvl = logging.DEBUG
console_log_lvl = logging.INFO
logger = logging.getLogger('iv')
logger.setLevel(console_log_lvl)
console = logging.StreamHandler(stream=sys.stdout)
console.setLevel(console_log_lvl)
formatter = logging.Formatter('%(asctime)s %(levelname)s <%(funcName)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
