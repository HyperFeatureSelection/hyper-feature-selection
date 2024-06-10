import logging

def configure_logger(name, verbosity=1, log_file="algorithm.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers will filter appropriately

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Console handler for optional console output based on verbosity
    if verbosity > 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel({
            0: logging.WARNING,  # Only WARNING and above
            1: logging.INFO,     # INFO and above
            2: logging.DEBUG     # DEBUG and above
        }.get(verbosity, logging.INFO))
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    return logger
