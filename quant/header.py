import logging as log

import tensorflow as tf
import numpy as np
import stock_data

SELL = 0
BUY = 1

log.basicConfig(filename='log/learn.log', filemode="w", level=log.INFO, format='%(message)s')