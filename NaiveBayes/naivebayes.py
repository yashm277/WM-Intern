import string
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math
import pickle
import os

nltk.download('stopwords')

# Load or process dataset
def load_data():
    df = pd.read_csv('data.csv')
    return df

def split_data_by_sentiment(data, sentiment):
    return data[data['sentiment'] == sentiment]['review'].tolist()

def cleaning(review):
    review = review.lower()
    review = review.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(review)
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words("english"))
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]
    return tokens

def default_value():
    return 1

def calculate_word_table(reviews):
    word_count = defaultdict(default_value)
    for review in reviews:
        result_string = ''.join(review)
        tokens = cleaning(result_string)
        for token in tokens:
            word_count[token]+=1
    return word_count

# Check if frequency tables exist, otherwise create them
if not (os.path.exists('positive_word_table.pkl') and os.path.exists('negative_word_table.pkl')):
    df = load_data()
    positive_data = split_data_by_sentiment(df, 'positive')
    negative_data = split_data_by_sentiment(df, 'negative')
    
    positive_word_table = calculate_word_table(positive_data)
    negative_word_table = calculate_word_table(negative_data)

    with open('positive_word_table.pkl', 'wb') as f:
        pickle.dump(positive_word_table, f)

    with open('negative_word_table.pkl', 'wb') as f:
        pickle.dump(negative_word_table, f)
else:
    with open('positive_word_table.pkl', 'rb') as f:
        positive_word_table = pickle.load(f)

    with open('negative_word_table.pkl', 'rb') as f:
        negative_word_table = pickle.load(f)

# Calculate the total counts and prior probabilities
df = load_data()
prior_positive = len(split_data_by_sentiment(df, 'positive')) / len(df) #0.9
prior_negative = len(split_data_by_sentiment(df, 'negative')) / len(df) #0.1 
# 1000 reviews, 900 pos, 100 neg
positive_sum = sum(positive_word_table.values()) # 50000
negative_sum = sum(negative_word_table.values()) # 10000

def sentiment_result(input_string, positive_word_table, negative_word_table, prior_positive, prior_negative, positive_sum, negative_sum):
    positive_sentiment, negative_sentiment = prior_positive, prior_negative
    input_tokens = cleaning(input_string)
    for token in input_tokens:
        positive_sentiment *= positive_word_table[token]/positive_sum  # 0.9 (prior) * 1/50 (movie) * 
        negative_sentiment *= negative_word_table[token]/negative_sum  # 0.1 * 1/10
    if positive_sentiment > negative_sentiment:
        print("positive review")
    else:
        print("negative review")

# Example usage
input_review = "movie movie movie"
sentiment_result(input_review, positive_word_table, negative_word_table, prior_positive, prior_negative, positive_sum, negative_sum)
