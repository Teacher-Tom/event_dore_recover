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

class OnlyBert(Model):
    '''
    只有bert编码与全连接的分类模型
    '''
    def __init__(self):
        super(OnlyBert, self).__init__()
        self.bertPre = hub.KerasLayer('bert_zh_preprocess', input_shape=(MAX_SEQ_LEN,))  # bert预处理层，对输入的文本数组转换为三个向量
        self.bertEncoder = hub.KerasLayer("bert_zh_L-12_H-768_A-12_4", trainable=False)  # 利用预训练bert对输入进行embedding
        self.dense = Dense(TAG_LEN,activation='softmax',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01))

    def call(self, x):
        print('inputs:',x.shape)
        inputs = self.bertPre(x)
        bert_out = self.bertEncoder(inputs)
        pooled_out = bert_out['pooled_output']
        out = self.dense(pooled_out)
        return out

if __name__ == '__main__':
    checkpoint_save_path = './checkpoint/onlybert_cls.ckpt'

    x_train, y_train = data_convert(TRAIN_PATH)
    x_val, y_val = data_convert(DEV_PATH)
    x_test, y_test = data_convert(TEST_PATH)

    model = OnlyBert()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5)  # 当loss不再下降自动结束训练
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                  save_weights_only=True,
                                                  save_best_only=True, verbose=1)  # 设置断点保存
    if os.path.exists(checkpoint_save_path+'.index' ):
        print('--------------读取上一次训练记录--------------')
        model.load_weights(checkpoint_save_path)
    print('y_train:', y_train.shape)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, embeddings_freq=1)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-1),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

    history = model.fit(x_train, y_train,
                        batch_size=16, epochs=1000,
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


