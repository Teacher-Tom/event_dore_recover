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
from Mixed_classification import *
# 修改tensorflow的warning级别使其不输出
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if __name__ == '__main__':
    att_dim = 64  # 注意力网络的神经元数
    units_a = 256
    units_b = 256
    batch_size = 16
    checkpoint_save_path = './checkpoint/bert_bilstm_crf_attdim{}.ckpt'.format(att_dim)  # 训练完成的模型保存位置
    loss = 'focal'  # normal
    checkpoint_save_path_mix = './checkpoint/cls_mix_ua{}_ub{}_batch{}.ckpt'.format(units_a, units_b, batch_size)
    model_mix = build_model(output_unit_a=units_a, output_unit_b=units_b, att_dim=64)
    model = BiLSTM_Att(att_dim)
    x_test, y_test = data_convert(TEST_PATH)
    x_test_en = data_enhance(x_test, units=units_b)
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------读取上一次训练记录1--------------')
        model.load_weights(checkpoint_save_path)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6),
                      loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=['categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
        score = model.evaluate(x_test, y_test)
        print('-------------评测结果1------------------')
        print('test_loss:', score[0])
        print('test_accuracy:', score[1])
        print('test_precision:', score[2])
        print('test_recall:', score[3])
        f1 = 2 * ((score[2] * score[3]) / (score[2] + score[3] + K.epsilon()))
        print('test_f1score:', f1)
    else:
        print('没有找到训练好的模型')


    if os.path.exists(checkpoint_save_path_mix + '.index'):
        print('--------------读取上一次训练记录2--------------')
        model_mix.load_weights(checkpoint_save_path_mix)
        model_mix.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                      loss=[multi_category_focal_loss1(get_alpha(), 2)],
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
        print('-------------评测结果2------------------')
        score = model_mix.evaluate([x_test, x_test_en], y_test)
        print(score)
        print('test_loss:', score[0])
        print('test_accuracy:', score[1])
        print('test_precision:', score[2])
        print('test_recall:', score[3])
        f1 = 2 * ((score[2] * score[3]) / (score[2] + score[3] + K.epsilon()))
        print('test_f1score:', f1)
    else:
        print('没有找到训练好的模型')








