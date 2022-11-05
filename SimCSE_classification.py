#-*- coding : utf-8-*-
# coding:unicode_escape
import math
import keras.layers
from keras.models import Model
from keras.layers import Layer
from keras.layers import Bidirectional, LSTM, Dense, Input, TimeDistributed, Activation, Dropout,GlobalAveragePooling1D,Multiply
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.utils import Sequence
from utils import *
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
from keras.models import Sequential
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import warnings
warnings.filterwarnings('ignore')
def data_enhance(x,units = 256):
    '''
    使用预训练的simCSE进行对比学习数据增强
    :param x:
    :return:
    '''
    model_path = './checkpoint/simCSE_u{}.ckpt'.format(units)
    model = SimCSE(output_units=units)
    model.load_weights(model_path)
    pre_x = model.predict(x)
    return pre_x

if __name__ == '__main__':
    units = 256
    checkpoint_save_path = './checkpoint/simcse_cls_u{}.ckpt'.format(units)
    model = Sequential([Dense(TAG_LEN,input_shape=(units,),activation='softmax',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01)),])
    x_train, y_train = data_convert(TRAIN_PATH)
    x_val, y_val = data_convert(DEV_PATH)
    x_test, y_test = data_convert(TEST_PATH)

    x_train = data_enhance(x_train)
    x_val = data_enhance(x_val)
    x_test = data_enhance(x_test)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=10)  # 当loss不再下降自动结束训练
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                  save_weights_only=True,
                                                  save_best_only=True, verbose=1)  # 设置断点保存
    if os.path.exists(checkpoint_save_path+'.index' ):
        print('--------------读取上一次训练记录--------------')
        model.load_weights(checkpoint_save_path)
    print('y_train:', y_train.shape)
    tensorboard = TensorBoard(log_dir='./logs/cls', histogram_freq=1, write_graph=False)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

    history = model.fit(x_train, y_train,
                        batch_size=16, epochs=15000,initial_epoch=0,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping, cp_callback,tensorboard])
    model.summary()
    print('-------------评测结果------------------')
    score = model.evaluate(x_test, y_test)
    print(score)
    print('test_loss:', score[0])
    print('test_accuracy:', score[1])
    print('test_precision:', score[2])
    print('test_recall:', score[3])
    f1 = 2 * ((score[2] * score[3]) / (score[2] + score[3] + K.epsilon()))
    print('test_f1score:', f1)


