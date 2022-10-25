# 工具类
import numpy as np
import pandas as pd
import keras.utils
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
import bert
from config import *

def data_convert(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')     # 读取数据集
    sentences = data['sentence'].values     # 读取所有句子
    tags = data['event_type'].values     # 读取所有事件类型tag id
    x, y = [], []
    for s in sentences:
        x.append(s)   # 将编码后的句子存入x数组中
    x = np.array(x)     # 将x转换为numpy数组
    for t in tags:
        label = keras.utils.to_categorical(t,TAG_LEN)   # Converts a class vector (integers) to binary class matrix.
        y.append(label)
    y = np.array(y)
    return x, y

def tokenizeAndEncode(tokenizer, text): # 使用bert将文本分词并编码为id
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

if __name__ == '__main__':
    BertTokenizer = bert.bert_tokenization.FullTokenizer    # 暂时用bert替代albert
    bert_layer = hub.KerasLayer("bert_zh_L-12_H-768_A-12_4",
                                trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    # 测试数据集编码
    train_x, train_y = data_convert(tokenizer, TRAIN_PATH)
    print(train_x)
    print(train_y)

