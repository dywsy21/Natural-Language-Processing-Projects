import jieba
import json
import logging
import os
from gensim.models import Word2Vec
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

def load_data(file_path):
    """Load data from JSON file - each line is a separate JSON object"""
    print(f"Loading data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSON data"):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue
    return data

def preprocess_text(text):
    """Segment Chinese text using jieba"""
    if not text or not isinstance(text, str):
        return []
    return list(jieba.cut(text))

class BaikeQACorpus:
    """Iterator for the Baike QA dataset"""
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        for item in tqdm(self.data, desc="Processing sentences"):
            if 'title' in item and item['title']:
                yield preprocess_text(item['title'])
            
            if 'desc' in item and item['desc']:
                yield preprocess_text(item['desc'])
            
            if 'answer' in item and item['answer']:
                yield preprocess_text(item['answer'])

def main():
    data_file = 'training_data/baike_qa_train.json'
    
    if not os.path.exists(data_file):
        print(f"Error: File {data_file} not found.")
        return
    
    data = load_data(data_file)
    print(f"Loaded {len(data)} QA pairs")
    
    os.makedirs('training_result', exist_ok=True)
    
    corpus = BaikeQACorpus(data)
    
    print("Training Word2Vec model...")
    model = Word2Vec(
        corpus, 
        vector_size=200,  # Dimension of word vectors
        window=5,         # Context window size
        min_count=5,      # Ignore words with fewer occurrences
        workers=32,       # Number of threads
        sg=1,             # Use skip-gram model
        epochs=5          # Number of training epochs
    )
    
    model_path = 'training_result/baike_word2vec.model'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    print("\nExample usage:")
    try:
        similar_words = model.wv.most_similar('地球', topn=5)
        print(f"Words most similar to '地球': {similar_words}")
    except KeyError:
        print("Word '地球' not in vocabulary. Try another word that appears in the corpus.")
    
    try:
        vector = model.wv['重力']
        print(f"Vector for '重力' has shape: {vector.shape}")
    except KeyError:
        print("Word '重力' not in vocabulary. Try another word that appears in the corpus.")
    
    print(f"Vocabulary size: {len(model.wv)}")

if __name__ == "__main__":
    main()
