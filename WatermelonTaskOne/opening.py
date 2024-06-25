import string
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math
import pickle

nltk.download('stopwords')

df = pd.read_csv('data.csv')

def split_data_by_sentiment(data, sentiment):
    return data[data['sentiment'] == sentiment]['review'].tolist()

df = pd.read_csv('data.csv')

def split_data_by_sentiment(data, sentiment):
    return data[data['sentiment'] == sentiment]['review'].tolist()

positive_data = split_data_by_sentiment(df, 'positive')
negative_data = split_data_by_sentiment(df, 'negative')
prior_positive = len(positive_data)/len(df)
prior_negative = len(negative_data)/len(df)

# this is the function for preprocessing the data
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

positive_word_table =  calculate_word_table(positive_data)
positive_sum=1
for key,value in positive_word_table.items():
    positive_sum+=(value-1)
negative_word_table =  calculate_word_table(negative_data)
negative_sum=1
for key,value in negative_word_table.items():
    negative_sum+=(value-1)

input_review = "The movie was amazing I had a great time watching it and would recommend it to anyone."

def sentiment_result(input_string, positive_word_table, negative_word_table, prior_positive, prior_negative, positive_sum, negative_sum):
    positive_sentiment,negative_sentiment = prior_positive, prior_negative
    input_tokens = cleaning(input_string)
    for token in input_tokens:
        positive_sentiment *= positive_word_table[token]/positive_sum
        negative_sentiment *= negative_word_table[token]/negative_sum
    if positive_sentiment>negative_sentiment:
        print("positive review")
    else:
        print("negative review")

with open('positive_word_table.pkl', 'wb') as f:
    pickle.dump(positive_word_table, f)

with open('negative_word_table.pkl', 'wb') as f:
    pickle.dump(negative_word_table, f)


sentiment_result(input_review, positive_word_table, negative_word_table, prior_positive, prior_negative, positive_sum, negative_sum)
    

