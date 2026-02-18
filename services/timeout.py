import logging
import time

def wait(logger: logging.Logger, timeout: float):
    """
    Waits until timeout occurs. Logs exception while waiting until timeout occurs.

    Args:
        logger (logging.Logger): Logger to wait until timeout occurs.
        timeout (float): Timeout in seconds.
    """
    logger.info(f" Wait for {timeout} seconds...\n")
    time.sleep(timeout)