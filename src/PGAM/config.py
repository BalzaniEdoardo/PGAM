from contextlib import contextmanager

class config:
    DEBUG = False

@contextmanager
def set_debug(value=True):
    old_value = config.DEBUG
    config.DEBUG = value
    try:
        yield
    finally:
        config.DEBUG = old_value
