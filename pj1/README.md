# Word2Vec Training on Baike QA Dataset

This project trains a Word2Vec model on the Baike QA dataset, which contains question-answer pairs in Chinese.

## Requirements

- Python 3.x
- jieba (for Chinese word segmentation)
- gensim (for Word2Vec implementation)
- tqdm (for progress visualization)

Install requirements using:

```bash
pip install -r requirements.txt
```

## File Structure

```
pj1/
├── main.py           # Main script for training the Word2Vec model
├── use_model.py      # Script for using the trained model
├── requirements.txt  # Project dependencies
└── training_result/  # Directory for storing trained models
    ├── baike_word2vec.model      # Saved model
    └── baike_word_vectors.txt    # Word vectors in text format
```

## Training

Run the training script:

```bash
python train_model.py
```

This will:
1. Load and process the data from `baike_qa_train.json`
2. Create a Word2Vec model with:
   - 200-dimensional word vectors
   - Context window size of 5
   - Minimum word frequency of 5
   - Skip-gram architecture (more accurate for infrequent words)
   - 5 training epochs
3. Save the trained model as `training_result/baike_word2vec.model`
4. Save the word vectors in text format as `training_result/baike_word_vectors.txt`

## Using the Trained Model

After training, you can use the interactive script to explore the model:

```bash
python use_model.py
```

The script allows you to:
- Find similar words to a given query word
- Compute similarity between two words
- View word segmentation for complex queries

Or programmatically:

- Find similar words:
  ```python
  similar_words = model.wv.most_similar('地球', topn=5)
  ```

- Get word vectors:
  ```python
  vector = model.wv['重力']
  ```

- Check word similarity:
  ```python
  similarity = model.wv.similarity('地球', '行星')
  ```

## Model Parameters

- `vector_size`: 200 (dimension of word vectors)
- `window`: 5 (context window size)
- `min_count`: 5 (minimum word frequency)
- `sg`: 1 (skip-gram model instead of CBOW)
- `workers`: 4 (number of threads)
- `epochs`: 5 (number of training iterations)
