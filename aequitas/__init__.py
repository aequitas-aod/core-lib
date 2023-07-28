import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('aequitas')

# let this be the last line of this file
logger.debug("Module %s correctly loaded", __name__)
