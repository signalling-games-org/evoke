# Utility functions for the Evoke package

import logging

class Colors:
    """
    Define ANSI escape codes for colors when printing to terminal.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Reset to default color
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger():

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    def debug(self, message):
        logging.debug(Colors.OKCYAN + message + Colors.ENDC)
    
    def info(self, message):
        logging.info(Colors.OKGREEN + message + Colors.ENDC)

    def warning(self, message):
        logging.warning(Colors.WARNING + message + Colors.ENDC)

    def error(self, message):
        logging.error(Colors.FAIL + message + Colors.ENDC)

    def critical(self, message):
        logging.critical(Colors.FAIL + Colors.BOLD + Colors.UNDERLINE + message + Colors.ENDC)

logger = Logger()