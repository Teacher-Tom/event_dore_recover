# 测试模型效果
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
import bert
import utils
from models import *
import keras.optimizers
import keras.losses
from utils import *
import os
import keras.callbacks
# 修改tensorflow的warning级别使其不输出
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if __name__ == '__main__':
    print('----------预测开始--------------')
    att_dim = 64  # 注意力网络的神经元数
    checkpoint_save_path = './checkpoint/bert_bilstm_crf_attdim{}.ckpt'.format(att_dim)  # 训练完成的模型保存位置

    model = BiLSTM_Att(att_dim)
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------读取上一次训练记录--------------')
        model.load_weights(checkpoint_save_path)
    else:
        print('没有找到训练好的模型')
        exit()
    while True:
        sentence = input('输入要预测的事件句子:')
        sent = [sentence]
        pred = model(sent)
        result = tf.argmax(pred,axis=1)[0].numpy()
        print('结果:',result,id_to_event_type[result])