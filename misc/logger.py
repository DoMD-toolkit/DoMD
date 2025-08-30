import logging

logger_format = '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s'
logging.basicConfig(format=logger_format, level=logging.INFO)


def get_logger(name):
    return logging.getLogger(name)


class DuplicateFilter:
    """
    Filters away duplicate log messages.
    Modified version of: https://stackoverflow.com/a/31953563/965332
    """

    def __init__(self, logger):
        self.msgs = set()
        self.logger = logger

    def filter(self, record):
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):
        if len(self.logger.filters) == 0:
            self.logger.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Example usage:
logger = get_logger(__name__)
logger.setLevel('INFO')
#logger.setLevel('WARNING')
