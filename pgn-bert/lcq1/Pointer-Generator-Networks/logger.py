import os
import logging


def init_logger(config):
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logfilename = config['filename'] + '.log'

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M:%S"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[fh, sh])
