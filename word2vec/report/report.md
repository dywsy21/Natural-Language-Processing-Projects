---
title: "Word2Vec 实验报告"
author: "王思宇"
date: \today
using_title: true
using_table_of_content: true
---

# Word2Vec 实验报告

**本次实验的完整源代码以及训练出来的模型见Github链接：<br>[https://github.com/dywsy21/Natural-Language-Processing-Projects](https://github.com/dywsy21/Natural-Language-Processing-Projects)。**

## 训练过程

训练过程主要分为如下几个部分：

### 读取训练数据

> 对应train_model.py的`load_data()`。

使用python的json库读取训练数据，并时刻监测数据是否有问题。本次实验提供的数据完整性很高，没有缺键少值等等这类问题，因此读取没有什么问题。

### 对训练数据进行分词

> 对应train_model.py的`preprocess_text()`。

直接使用结巴分词库`(jieba)`来对训练数据进行分词。分出来的一个个单词的全体构成了训练出的词向量的Vocabulary.

### 开始训练词向量

> 对应train_model.py的`Word2Vec()`。

在训练之前，为训练数据套了一层class，使用`tqdm`库实现了训练时加上进度条。这样可以方便地查看训练进展。

训练时设置的参数如下：

```python
model = Word2Vec(
        corpus, 
        vector_size=200,  # Dimension of word vectors
        window=5,         # Context window size
        min_count=5,      # Ignore words with fewer occurrences
        workers=32,       # Number of threads
        sg=1,             # Use skip-gram model
        epochs=5          # Number of training epochs
    )
```

训练持续了一个多小时，最终输出的信息如下：

```
2025-03-20 12:11:25,678 : INFO : EPOCH 4: training on 320363372 raw words (248748587 effective words) took 808.6s, 307617 effective words/s

2025-03-20 12:11:25,678 : INFO : Word2Vec lifecycle event {'msg': 'training on 1601816860 raw words (1243707899 effective words) took 4044.4s, 307513 effective words/s', 'datetime': '2025-03-20T12:11:25.678235', 'gensim': '4.3.3', 'python': '3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)]', 'platform': 'Windows-11-10.0.22631-SP0', 'event': 'train'}

2025-03-20 12:11:25,678 : INFO : Word2Vec lifecycle event {'params': 'Word2Vec<vocab=500467, vector_size=200, alpha=0.025>', 'datetime': '2025-03-20T12:11:25.678235', 'gensim': '4.3.3', 'python': '3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)]', 'platform': 'Windows-11-10.0.22631-SP0', 'event': 'created'}    
```

可以看到训练数据集一共有约1e9个有效单词，训练出的模型的Vocabulary有500467个单词。

## 使用训练出的词向量模型

我们编写了示例代码展示如何使用训练出的词向量，源代码为`use_model.py`。

该交互式程序提供了两种主要功能：

1. 查找相似词：输入一个单词，程序会返回词向量空间中最相似的10个词及其相似度分数。
2. 计算词语相似度：输入两个词（用空格分隔），程序会计算它们在词向量空间中的余弦相似度。

### 模型加载

程序首先从保存的模型文件加载训练好的Word2Vec模型：

```python
def load_model(model_path):
    """Load a trained Word2Vec model"""
    print(f"Loading model from {model_path}")
    return Word2Vec.load(model_path)
```

### 交互式查询

程序提供了一个交互式界面，允许用户进行持续的查询：

- 输入单个词，获取与之最相似的10个词
- 输入两个词（空格分隔），计算它们之间的相似度
- 输入'q'退出程序

### 词汇分割处理

当用户查询的词不在词汇表中时，程序会尝试使用jieba分词对其进行分割，并提示用户尝试查询分割后的单个词：

```python
segments = segment_text(query)
if len(segments) > 1:
    print(f"The word might need segmentation. Segmented as: {' '.join(segments)}")
    print("Try searching for one of these segments.")
```

### 使用示例

以下是程序运行时的交互示例：

1. 查找相似词：
```
    Query: 地球
    Words most similar to '地球':
    月球: 0.8479
    太阳系: 0.8038
    地球表面: 0.7788
    星球: 0.7694
    自转: 0.7640
    天体: 0.7599
    星体: 0.7494
    恒星: 0.7469
    公转: 0.7465
    银河系: 0.7459
```

2. 计算词语相似度：
```
    Query: 猫 狗
    Similarity between '猫' and '狗': 0.7857
```

## 总结

本次实验成功训练了一个基于中文百科数据的Word2Vec模型，通过该模型可以有效地捕获词语之间的语义关系。我们训练出的模型包含500,467个词汇，每个词向量维度为200，使用Skip-gram模型和多线程处理，在32个工作线程下对超过16亿原始词（约12.4亿有效词）的训练仅用了约4000秒，每秒处理超过30万有效词。通过交互式查询功能验证，模型能够有效支持词语相似度计算和近义词查找等应用，为下游NLP任务如文本分类、情感分析等提供了良好的语义特征基础。
