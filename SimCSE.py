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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def transfer_data(path, seed=114514):
    '''
    将数据集shuffle后并将每一句话复制一份
    :param path:
    :return:
    '''

    # shuffle
    x,y = utils.data_convert(path)
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    tf.random.set_seed(seed)
    # 复制每一句
    rx = []
    ry = []
    for s in x:
        rx.append(s)
        rx.append(s)
    for t in y:
        ry.append(t)
        ry.append(t)

    return np.array(rx), np.array(ry)



class CseGenerator(Sequence):
    def __init__(self,x_set,y_set,batch_size,shuffle=True):
        self.x,self.y = x_set,y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bertPre = hub.KerasLayer('bert_zh_preprocess', input_shape=(MAX_SEQ_LEN,))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        print('idx:',idx)
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        pre_x = []
        pre_y = []
        for s in batch_x:
            pre_x.append(s)
            pre_x.append(s)
        for t in batch_y:
            pre_y.append(t)
            pre_y.append(t)
        return np.array(pre_x),np.array(pre_y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



def simcse_loss(y_true, y_pred):
    '''
    这个loss的输入中y_true只是凑数的，并不起作用。
    因为真正的y_true是通过batch内数据计算得出的。
    y_pred就是batch内的每句话的embedding，通过bert编码得来。
    '''
    # print('y_true:',y_true,'y_pred:',y_pred)
    idxs = K.arange(0, K.shape(y_pred)[0])  # 这行的作用，就是生成batch内句子的编码。根据我们的例子，idxs就是：[0,1,2,3,4,5]
    # print('idxs:',idxs)
    idxs_1 = idxs[None, :]  # 给idxs添加一个维度，变成： [[0,1,2,3,4,5]]
    # print('idxs1:',idxs_1)
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None] # 这个其实就是生成batch内每句话同义的句子的id。
    # print('idxs2:',idxs_2)
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    # print('loss:',loss)
    return K.mean(loss)

def simcse_loss2(y_true, y_pred):
    """
    simcse loss
    """
    idxs = tf.range(0, tf.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = tf.equal(idxs_1, idxs_2)
    y_true = tf.cast(y_true, tf.keras.backend.floatx())
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
    similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
    similarities = similarities / 0.05
    loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
    return tf.reduce_mean(loss)

class SimCSE(Model):
    def __init__(self,drop_rate=0.1,output_units=128,activation='tanh'):
        super(SimCSE, self).__init__()
        self.bertPre = hub.KerasLayer('bert_zh_preprocess', input_shape=(MAX_SEQ_LEN,))  # bert预处理层，对输入的文本数组转换为三个向量
        self.bertEncoder = hub.KerasLayer("bert_zh_L-12_H-768_A-12_4", trainable=True)  # 利用预训练bert对输入进行embedding
        self.dense = Dense(output_units,activation=activation,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01))


    def call(self, x):
        print('inputs:',x.shape)
        inputs = self.bertPre(x)
        bert_out = self.bertEncoder(inputs)
        pooled_out = bert_out['pooled_output']
        print('pooled:',pooled_out.shape)
        out = self.dense(pooled_out)
        print('out.shape:', out.shape)
        return out

if __name__ == '__main__':
    batch_size = 16
    units = 256
    x, y = utils.data_convert(TRAIN_PATH)
    print(x,y)
    checkpoint_save_path = './checkpoint/simCSE_u{}.ckpt'.format(units)
    model = SimCSE(output_units=units)
    tensorboard = TensorBoard(log_dir='./logs/simcse', histogram_freq=1, embeddings_freq=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss',
                                                   patience=5)   # 当loss不再下降自动结束训练
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,save_weights_only=True,verbose=1)    # 设置断点保存
    if os.path.exists(checkpoint_save_path+'.index'):
        print('--------------读取上一次训练记录--------------')
        model.load_weights(checkpoint_save_path)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss= simcse_loss2)
    generator = CseGenerator(x,y,batch_size=batch_size)
    model.fit(generator,epochs=15,initial_epoch=0,callbacks=[cp_callback,tensorboard,early_stopping])
    model.summary()


