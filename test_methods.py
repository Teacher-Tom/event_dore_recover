import numpy as np
import tensorflow as tf
import keras.backend as K
import pandas as pd

types = [[1.] for i in range(33)]
types[32][0] = 0.3  # transport
types[0][0] = 0.6  # attack
types[21][0] = 0.8  # meet
types[1][0] = 0.9  # die
types[30][0] = 3.  # execute
types[12][0] = 3.  # acquit
print(types)
types = np.array(types)
types = tf.constant(types,dtype=tf.float32)
print(types)