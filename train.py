# 训练模型
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
from matplotlib import pyplot as plt
import time
import keras.metrics
import keras_metrics as km
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()  ##.model
        val_targ = self.validation_data[1]  ###.model
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average=None)  ###
        _val_precision = precision_score(val_targ, val_predict, average=None)  ###
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        # print("— val_f1: %f — val_precision: %f — val_recall: %f" %(_val_f1, _val_precision, _val_recall))
        print("— val_f1: %f " % _val_f1)
        return



def createEncoder():    # 创建bert编码器
    BertTokenizer = bert.bert_tokenization.FullTokenizer    # 暂时用bert替代albert
    bert_layer = hub.KerasLayer("bert_zh_L-12_H-768_A-12_4",
                                trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    return tokenizer

def trainAndEvaluate(train_data_path, dev_data_path):   # 训练模型并验证效果,并保存模型
    encoder = createEncoder()
    train_x, train_y = utils.data_convert(encoder,train_data_path)
    dev_x, dev_y = utils.data_convert(encoder,dev_data_path)
    # TODO

if __name__ == '__main__':
    print('----------训练开始--------------')
    att_dim = 64    # 注意力网络的神经元数
    checkpoint_save_path = './checkpoint/bert_bilstm_crf_attdim{}.ckpt'.format(att_dim)  # 训练完成的模型保存位置
    x_trian,y_train = data_convert(TRAIN_PATH)
    print(x_trian,y_train)
    x_val, y_val = data_convert(DEV_PATH)
    x_test, y_test = data_convert(TEST_PATH)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=10)   # 当loss不再下降自动结束训练
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                    save_weights_only=True,
                                    save_best_only=True,verbose=1)    # 设置断点保存
    tensorboard = TensorBoard(log_dir='./logs/origin', histogram_freq=1, embeddings_freq=0,write_graph=False)
    model = BiLSTM_Att(att_dim)
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------读取上一次训练记录--------------')
        model.load_weights(checkpoint_save_path)
    print('y_train:', y_train.shape)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss= keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['categorical_accuracy',keras.metrics.Precision(),keras.metrics.Recall()])

    history = model.fit(x_trian,y_train,
                        batch_size=16,epochs=140,initial_epoch=120,
                        validation_data=(x_test,y_test),
                        callbacks=[early_stopping,cp_callback,tensorboard])
    model.summary()
    print('-------------评测结果------------------')
    score = model.evaluate(x_test,y_test)
    print(score)
    print('test_loss:',score[0])
    print('test_accuracy:',score[1])
    print('test_precision:',score[2])
    print('test_recall:',score[3])
    f1 = 2 * ((score[2]*score[3])/(score[2]+score[3]+K.epsilon()))
    print('test_f1score:',f1)

    # 绘制学习曲线
    his = history.history
    acc = his['categorical_accuracy']
    val_acc = his['val_categorical_accuracy']
    loss = his['loss']
    val_loss = his['val_loss']

    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(acc,label='Training Acc')
    plt.plot(val_acc,label='Validation Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(loss,label='Training Loss')
    plt.plot(val_loss,label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.savefig('./training_diagrams/bert_bilstm_attention_attdim{}_{}_{}.png'.format(att_dim,time.localtime().tm_mon,time.localtime().tm_mday))
    plt.show()
