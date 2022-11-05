# 存放各类自定义模型和层
import keras.layers
from keras.models import Model
from keras.layers import Layer
from keras.layers import Bidirectional, LSTM, Dense, Input, TimeDistributed, Activation, Dropout,GlobalAveragePooling1D,Multiply
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from config import *
import numpy as np
import tensorflow_text as text
import keras.backend as K
from keras.layers.core import Lambda
import tensorflow_addons as tfa
from keras import regularizers

def Att(att_dim,inputs,name):
    V = inputs
    QK = Dense(att_dim,bias=None)(inputs)
    QK = Activation("softmax",name=name)(QK)
    MV = Multiply()([V, QK])
    return(MV)

class BiLSTM_Att(Model):
    def __init__(self,att_dim):
        super(BiLSTM_Att, self).__init__()
        self.bertPre = hub.KerasLayer('bert_zh_preprocess',input_shape=(MAX_SEQ_LEN,)) # bert预处理层，对输入的文本数组转换为三个向量
        self.bertEncoder = hub.KerasLayer("bert_zh_L-12_H-768_A-12_4",trainable=True)   # 利用预训练bert对输入进行embedding
        self.bilstm = Bidirectional(LSTM(units=128,return_sequences=True),input_shape=(),name='bi-lstm')
        self.dropout = Dropout(0.5,name='dropout')
        self.att_dense1 = Dense(att_dim,use_bias=True,activation='tanh',
                                kernel_regularizer=regularizers.l2(0.01),
                                bias_regularizer=regularizers.l2(0.01))
        self.att_dense2 = Dense(256,use_bias=False,activation='softmax',
                                kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1_l2(0.01))
        self.dropout2 = Dropout(0.5,name='dropout2')
        self.dense = Dense(TAG_LEN,activation='softmax',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01))
    def call(self,x):
        print('input:',x)
        x = self.bertPre(x) # 预处理语句，将字符序列转换为三个编码后的向量
        x = self.bertEncoder(x) # 对序列做embedding
        x = x['sequence_output']    # 只需要每个token的embedding
        x = self.bilstm(x)
        v = self.dropout(x)
        # print('v.shape:',v.shape)
        qk = self.att_dense1(v) # 注意力层的第一个全连接层，相当于H = K.tanh(K.dot(inputs, self.W) + self.b)
        score_mv = self.att_dense2(qk)  # 注意力层的第二个全连接层,相当于K.softmax(K.dot(H, self.V), axis=1)
        out = K.sum(score_mv,axis=1)    # 对一句话中所有token的分数求和
        out = self.dropout2(out)
        y = self.dense(out) # 全连接层，输出33种类型的概率值
        # print('y.shape:', y.shape)
        return y

if __name__ == '__main__':


    preprocessor = hub.load('bert_zh_preprocess')
    tokenizer = hub.KerasLayer(preprocessor.tokenize)
    special_tokens_dict = preprocessor.tokenize.get_special_tokens_dict()
    print(special_tokens_dict)
    encoder = hub.KerasLayer("bert_zh_L-12_H-768_A-12_4",trainable=True)
    encoder2 = hub.load("bert_zh_L-12_H-768_A-12_4")
    mlm = hub.KerasLayer(encoder2.mlm, trainable=True)
    text_input = keras.layers.Input(shape=(),dtype=tf.string)
    input = tokenizer(text_input)
    pooled_output = mlm(input)['sequence_output']
    model = keras.Model(text_input,pooled_output)
    sents = tf.constant(['大家好啊我是说的道理，今天给大家来点想看的东西啊。',
                         '大家好啊我是说的道理，今天给大家来点想看的东西啊。',
                         '大家好啊我是说的道理，今天给大家来点想看的东西啊。'])
    print(model(sents))
