
import logging
import logging.config

# Global configuration
from dqa.configurators import LOG_CONF


def make_logger(module):
    """
    Create and configure a logger for the specified module.

    Args:
        module (str): The name of the module.

    Returns:
        logging.Logger: The configured logger instance.
    """

    logging.config.dictConfig(LOG_CONF)
    logging.getLogger("py4j").setLevel(logging.ERROR)

    return logging.getLogger(module)
