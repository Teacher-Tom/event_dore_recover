# 初次复现论文中基于Bert+BiLSTM+Attention的事件类型检测——2022.10.25

因为原程序基于较老的tensorflow1版本，兼容性很差，于是在程永学长的程序基础上，复现并将模型迁移到了tensorflow2.9的平台上。其中有许多困难与注意点，以下将一一列举出：

## Bert

原模型在编码和嵌入层采用的是AlBert,即Bert的变形，但由于Albert基于tf1.0，与2.0不兼容，所以暂时无法在2.0中采用，故使用基本的bert模型代替。

另外在使用bert的过程中，有一些困难点：

1.我使用了tensorflow_hub中的bert_zh_L-12_H-768_A-12_4（第四版）模型，用于tokenize与embedding。起初使用了第二版模型，在输入数据时会报错，使用新版后就没问题了。另外建议下载到本地，联网可能会下载失败。bert需要有三个输入向量（inputIds、inputMask、segmentIds），而我们有的数据只是句子字符数组，而且每句话不等长，手动处理过于繁琐。利用hub官方提供的bert_zh_preprocess预处理层，可以直接将输入的文本转化为三个分词编码后的向量,随后可以直接输入bert模型。

2.需要注意bert模型的输出是一个字典，包含了不同类型的输出向量。其中最重要的有两个：pooled_output把整个句子当成整体做池化,shape=[batch_size, 768];sequence_output代表句子中每个分词，shape=[batch_size, seq_length, 768]

## Attention

基本是借鉴了学长的AttentionLayer，但是还没有完全搞懂这种Attention的原理，并且我用了两个全连接层来代替原来的矩阵乘法，我想大概效果是一样的？

### 双重注意力？局部/全局注意力？

我目前看了挺久学长的代码了，愣是没看出来哪里有双重注意力了，只有一个普通Attention以及self-Attention，而且也并没有将这两个结合起来。可能是我水平不够还没看出来？或者是真的没有？总之得去找学长问问。

ps：貌似只有一个Attention的效果也已经很接近论文里的了

## CRF

我一开始用了学长的bert+bilstm+crf的模型，但是搞了半天发现这好像并不是服务于我这个事件检测任务的，而是其他的序列标注任务。**那么为什么CRF不能用在这个分类问题呢？**毕竟我还不会crf的机理，希望以后能解答。

## 兼容性

**tensorflow的兼容性就是一坨答辩（无慈悲）**

首先1.0和2.0的差别很大，好多方法没法混用，基于1.0的模型也很可能没法在2.0上跑，虽然有tf.compact.v1能兼容部分方法，但依然会有奇奇怪怪的问题。总之尽量避免用一些老方法，老模型

还有keras.contrib里的layer居然不兼容tf2.0，真的是卡了我好久（吐了）,比如CRF层，现在请用Tensorflow_addon里的CRF替代。

## 训练结果

虽然目前用的是普通bert和单层attention，但是效果已经很不错了。

以下是不同attention_dim大小下的训练数据,学长用的是64，但我实际用下来是128更好。不过过拟合还是很严重，后续应该尝试正则化和减少参数数量。

dim=64

![bert_bilstm_attention_attdim64](D:\production_environment\python\nlp_projects\event_dore_recover\training_diagrams\bert_bilstm_attention_attdim64.png)

dim=128

![bert_bilstm_attention_attdim128](D:\production_environment\python\nlp_projects\event_dore_recover\training_diagrams\bert_bilstm_attention_attdim128.png)

dim=256

![bert_bilstm_attention_attdim256](D:\production_environment\python\nlp_projects\event_dore_recover\training_diagrams\bert_bilstm_attention_attdim256.png)

## 结语

这是我第一次自己用TensorFlow成功搭建并训练模型，意义非凡。总体看下来，其实要发表一篇论文也不是非常难，重要的是多看论文，多实验。