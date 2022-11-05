# 融合Bert-bilstm-att和对比学习模型
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
def data_enhance(x,units = 512):
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

def get_alpha():
    # 产生33个事件类别
    types = [[1.] for i in range(33)]
    types[32][0] = 0.4 # transport
    types[0][0] = 0.7  # attack
    types[21][0] =1  # meet
    types[1][0] = 1.1 # die
    types[30][0] = 5.   # execute
    types[12][0] = 5.   # acquit
    print(types)
    alpha = np.array(types)
    return alpha


def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    #alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed


def build_model(output_unit_a,output_unit_b,att_dim):
    input_bi = Input(type_spec=tf.RaggedTensorSpec(shape=[None,], dtype=tf.string))  # 原模型的输入
    input_cse = Input(shape=(output_unit_b,)) # 增强后的句子向量输入
    # 以下是bert+bilstm+att模型
    bertPre = hub.KerasLayer('bert_zh_preprocess', input_shape=(MAX_SEQ_LEN,))(input_bi)  # bert预处理层，对输入的文本数组转换为三个向量
    bertEncoder = hub.KerasLayer("bert_zh_L-12_H-768_A-12_4", trainable=True)(bertPre)  # 利用预训练bert对输入进行embedding
    bertEncoder = bertEncoder['sequence_output']
    bilstm = Bidirectional(LSTM(units=128, return_sequences=True), input_shape=(), name='bi-lstm')(bertEncoder)
    dropout = Dropout(0.5, name='dropout')(bilstm)
    att_dense1 = Dense(att_dim, use_bias=True, activation='tanh',
                            kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))(dropout)
    att_dense2 = Dense(output_unit_a, use_bias=False, activation='softmax',
                            kernel_regularizer=regularizers.l2(0.01),
                            activity_regularizer=regularizers.l1_l2(0.01))(att_dense1)
    out_a = K.sum(att_dense2, axis=1)
    # 融合模型输出和增强句子向量,选择方式[a;b]
    print('a.shape:',out_a.shape,'b.shape:',input_cse.shape)
    out_ab = tf.concat([out_a,input_cse],1)
    out_ab = Dropout(0.5, name='dropout2')(out_ab)
    print('out_ab.shape:',out_ab.shape)
    out = Dense(TAG_LEN, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                       bias_regularizer=regularizers.l2(0.01))(out_ab)

    model = Model(inputs=[input_bi,input_cse],outputs=out)
    return model


if __name__ == '__main__':
    units_a = 256
    units_b = 256
    batch_size = 16
    loss = 'focal'  # normal
    checkpoint_save_path = './checkpoint/cls_mix_ua{}_ub{}_batch{}_{}.ckpt'.format(units_a,units_b,batch_size,loss)
    model = build_model(output_unit_a=units_a, output_unit_b=units_b, att_dim=64)
    x_train, y_train = data_convert(TRAIN_PATH)
    x_val, y_val = data_convert(DEV_PATH)
    x_test, y_test = data_convert(TEST_PATH)

    x_train_en = data_enhance(x_train,units=units_b)
    x_val_en = data_enhance(x_val,units=units_b)
    x_test_en = data_enhance(x_test,units=units_b)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=10)  # 当loss不再下降自动结束训练
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                  save_weights_only=True,
                                                  save_best_only=True, verbose=1)  # 设置断点保存
    if os.path.exists(checkpoint_save_path+'.index' ):
        print('--------------读取上一次训练记录--------------')
        model.load_weights(checkpoint_save_path)
    tensorboard = TensorBoard(log_dir='./logs/mix_{}'.format(loss), histogram_freq=1,write_graph=False)
    '''
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    '''
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss=[multi_category_focal_loss1(get_alpha(),2)],
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

    history = model.fit([x_train,x_train_en], y_train,
                        batch_size=batch_size, epochs=170,initial_epoch=150,
                        validation_data=([x_test,x_test_en], y_test),
                        callbacks=[early_stopping, cp_callback,tensorboard])
    model.summary()
    print('-------------评测结果------------------')
    score = model.evaluate([x_test,x_test_en], y_test)
    print(score)
    print('test_loss:', score[0])
    print('test_accuracy:', score[1])
    print('test_precision:', score[2])
    print('test_recall:', score[3])
    f1 = 2 * ((score[2] * score[3]) / (score[2] + score[3] + K.epsilon()))
    print('test_f1score:', f1)


