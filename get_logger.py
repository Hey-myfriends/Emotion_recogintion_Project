import logging

def get_root_logger(level=logging.INFO, console_out=True, logName=None, fmt=None):
    logger = logging.getLogger("")
    logger.setLevel(level)
    if fmt is None:
        fmt = logging.Formatter(fmt="%(asctime)s - %(levelname)-8s - %(filename)-20s - Line:%(lineno)-4s - %(message)s")
    else:
        fmt = logging.Formatter(fmt=fmt)

    if console_out:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(sh)

    if logName is not None:
        fh = logging.FileHandler(logName, mode="w")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        logger.addHandler(fh)

    return logger

logger = get_root_logger(level=logging.INFO, console_out=True, logName="train_test.log")

if __name__ == "__main__":
    logger = get_root_logger(level=logging.DEBUG, console_out=True, logName="./log.log")
    logger.info("this is info")
    logger.debug("this is debug")
    logger.error("this is error")
    logger.warning("this is warning")