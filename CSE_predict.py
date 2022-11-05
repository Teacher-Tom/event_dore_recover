import math
import keras.layers
from keras.models import Model
from keras.layers import Layer
from keras.layers import Bidirectional, LSTM, Dense, Input, TimeDistributed, Activation, Dropout,GlobalAveragePooling1D,Multiply
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.utils import Sequence
import utils
from config import *
import numpy as np
import tensorflow_text as text
import keras.backend as K
from keras.layers.core import Lambda
import tensorflow_addons as tfa
from keras import regularizers
import os
import keras.callbacks
# 导入Tensorboard
from keras.callbacks import TensorBoard
from SimCSE import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
if __name__ == '__main__':
    checkpoint_save_path = './checkpoint/simCSE.ckpt'
    model = SimCSE()
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss',
                                                   patience=10)  # 当loss不再下降自动结束训练
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True)  # 设置断点保存
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------读取上一次训练记录--------------')
        model.load_weights(checkpoint_save_path)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6),
                  loss=simcse_loss2)
    y = model.predict(['任何邪恶，终将绳之以法','任何罪恶，终将绳之以法','到达世界最高城，理塘'])
    print(y)
