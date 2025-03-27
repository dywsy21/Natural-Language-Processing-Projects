from gensim.models import Word2Vec
import jieba
import os

def load_model(model_path):
    """Load a trained Word2Vec model"""
    print(f"Loading model from {model_path}")
    return Word2Vec.load(model_path)

def segment_text(text):
    """Segment Chinese text using jieba"""
    return list(jieba.cut(text))

def main():
    model_path = 'training_result/baike_word2vec.model'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Make sure to run main.py first to train the model.")
        return
    
    model = load_model(model_path)
    
    print(f"Model loaded with vocabulary size: {len(model.wv)}")
    print("\nEnter a word to find similar words, or 'q' to quit")
    print("Enter two words separated by a space to compute similarity")
    
    while True:
        query = input("\nQuery: ")
        
        if query.lower() == 'q':
            break
        
        if ' ' in query:
            words = query.split()
            if len(words) == 2:
                word1, word2 = words
                try:
                    similarity = model.wv.similarity(word1, word2)
                    print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
                except KeyError as e:
                    print(f"Error: {e}. One or both words not in vocabulary.")
            else:
                print("For similarity, enter exactly two words separated by a space.")
        else:
            try:
                similar_words = model.wv.most_similar(query, topn=10)
                print(f"Words most similar to '{query}':")
                for word, score in similar_words:
                    print(f"  {word}: {score:.4f}")
            except KeyError:
                print(f"Word '{query}' not in vocabulary.")
                segments = segment_text(query)
                if len(segments) > 1:
                    print(f"The word might need segmentation. Segmented as: {' '.join(segments)}")
                    print("Try searching for one of these segments.")

if __name__ == "__main__":
    main()
