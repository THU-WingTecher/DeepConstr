import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(name)s] - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + "[%(levelname)s]" + reset + "%(message)7s",
        logging.INFO: green + "[%(levelname)s]" + reset + "  %(message)7s",
        logging.WARNING: yellow + "[%(levelname)s]" + reset + "  %(message)7s",
        logging.ERROR: red + "[%(levelname)s]" + reset + "  %(message)7s",
        logging.CRITICAL: bold_red + "[%(levelname)s]" + reset + "%(message)7s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())


def init_logging(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


AUTOINF_LOG = init_logging("autoinf")
