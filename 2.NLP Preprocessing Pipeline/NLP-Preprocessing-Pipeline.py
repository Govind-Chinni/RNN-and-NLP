import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_nlp(sentence):
    # Step 1: Tokenize the sentence
    tokens = word_tokenize(sentence)
    print("Tokens:", tokens)
    
    # Step 2: Remove English stopwords
    stop_words = set(stopwords.words('english'))
    tokens_no_stopwords = [word for word in tokens if word.lower() not in stop_words]
    print("After Stopword Removal:", tokens_no_stopwords)
    
    # Step 3: Apply stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens_no_stopwords]
    print("After Stemming:", stemmed_tokens)

# Example sentence
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
preprocess_nlp(sentence)
