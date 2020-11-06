import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import json

# 1. News Classifier
json_lst = []
for line in open('news-dataset.json', 'r'):
    json_lst.append(json.loads(line))

df = pd.DataFrame(json_lst)
df = df.drop(columns=['link'])

POLITICS = ['POLITICS']
NEWS = ['BUSINESS', 'THE WORLDPOST', 'WORLD NEWS', 'WORLDPOST', 'MEDIA', 'IMPACT', 'WEIRD NEWS', 'GOOD NEWS', 'CRIME', 'GREEN', 'ENVIRONMENT', 'RELIGION', 'SCIENCE', 'TECH']
LIFE = ['WELLNESS', 'HEALTHY LIVING', 'FIFTY', 'TRAVEL', 'STYLE & BEAUTY', 'STYLE', 'TASTE', 'PARENTING', 'PARENTS', 'FOOD & DRINK', 'SPORTS', 'HOME & LIVING', 'WEDDINGS', 'DIVORCE', 'MONEY', 'EDUCATION', 'COLLEGE']
ENTERTAINMENT = ['ENTERTAINMENT', 'COMEDY', 'ARTS & CULTURE', 'ARTS', 'CULTURE & ARTS']
COMMUNITIES = ['WOMEN', 'LATINO VOICES', 'BLACK VOICES', 'QUEER VOICES']

# add new category variable to the dataset
df.loc[df.category.isin(POLITICS), 'new_category'] = 'POLITICS'
df.loc[df.category.isin(NEWS), 'new_category'] = 'NEWS'
df.loc[df.category.isin(LIFE), 'new_category'] = 'LIFE'
df.loc[df.category.isin(ENTERTAINMENT), 'new_category'] = 'ENTERTAINMENT'
df.loc[df.category.isin(COMMUNITIES), 'new_category'] = 'COMMUNITIES'

# define a function that combines the headline and the description
def headline_desc_mapping(headline, description):
    data = headline.map(str) + ". " + description.map(str)
    return data

headline = df.headline
desc = df.short_description

headline_and_desc = headline_desc_mapping(headline, desc)
category = df['new_category']

import spacy
nlp = spacy.load('en_core_web_sm')

def text_preprocessing(data):
    # remove unnecessary parts of text
    data = data.apply(lambda row: ' '.join(str(word) for word in nlp(row) if not (word.is_stop or word.is_punct or word.is_digit or word.is_space or word.is_quote or word.is_bracket or word.like_url)))
    # lowercase
    data = data.apply(lambda row: row.lower())
    # # lemmatize the sentence
    data = data.apply(lambda row: ' '.join(str(word.lemma_) for word in nlp(row)))
    return data

# headline_and_desc = headline_and_desc.astype('str')
# headline_and_desc = text_preprocessing(headline_and_desc)

# label encode the target variable
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
lb.fit(category)
category_encoded = lb.transform(category)

# one hot encoding the target variable
# from keras.utils import np_utils
# category_onehot = np_utils.to_categorical(category_encoded)

# # text Vectorization

# import tensorflow as tf 
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# vocab_size=78000
# vectorizer = TextVectorization(max_tokens=vocab_size)
# vectorizer.adapt(headline_and_desc.values)

# # Split the dataset
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(headline_and_desc, category_onehot, test_size=0.3, random_state=42, shuffle=True)

# Text Classifier: Bidirectional LSTM

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import LSTM, Bidirectional
# from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten
# from tensorflow.keras import optimizers, losses, metrics
# from tensorflow.keras.callbacks import EarlyStopping

# model = tf.keras.Sequential([
#     vectorizer,
#     tf.keras.layers.Embedding(
#         input_dim=len(vectorizer.get_vocabulary()),
#         output_dim=128),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(len(np.unique(category_encoded)), activation='softmax')
# ])

# model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.categorical_crossentropy, metrics=['acc'])

# epochs=3
# batch_size=128
# # es = EarlyStopping(monitor='val_acc', patience=10)

# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)

# define a function that inversers the prediction values to the class

def inverse_prediction(prediction):
    prediction_class = [np.argmax(i, axis=None, out=None) for i in prediction]
    prediction_label = lb.inverse_transform(prediction_class)
    return prediction_label

# 2. News Recommender

df = pd.concat([df, headline_and_desc], axis=1)
df = df.rename(columns={0: 'headline_and_desc'})
df.headline_and_desc = df.headline_and_desc.replace('', np.nan)
df.dropna(inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

# define a function to compute the euclidean similarity

# 2.1. tfidf: headline_and_desc
def headline_and_desc_tfidf_recommender(df, news, sample_text, num_similar_items):
    tfidf = TfidfVectorizer(min_df=0)
    tfidf.fit(news)
    news_vec = tfidf.transform(news)
    sample_vec = tfidf.transform(sample_text)

    distance = pairwise_distances(news_vec, sample_vec)
    indices = np.argsort(distance.ravel())[0:num_similar_items]
    news_list = pd.DataFrame({
        'headline': df['headline'][indices],
        'category': df['new_category'][indices]
    })
    result = news_list.to_dict('records')
    return result

# 2.2. tfidf: headline_and_desc and category
from sklearn.preprocessing import OneHotEncoder

# define a function to compute the euclidean similarity

def text_and_category_tfidf_recommender(df, news, sample_text, category, sample_category, num_similar_items, w1, w2):
    # tfidf vectorize text
    tfidf = TfidfVectorizer(min_df=0)
    tfidf.fit(news)
    news_vec = tfidf.transform(news)
    sample_vec = tfidf.transform(sample_text)

    # one-hot-encode the category
    onehotencoder = OneHotEncoder()

    category = category.values.reshape(-1,1)
    sample_category = sample_category.reshape(-1,1)

    onehotencoder.fit(category)
    category_onehot = onehotencoder.transform(category)
    sample_category_onehot = onehotencoder.transform(sample_category)

    # compute the euclidean similarity
    news_distance = pairwise_distances(news_vec, sample_vec)
    category_distance = pairwise_distances(category_onehot, sample_category_onehot)
    weighted_distance = (w1 * news_distance + w2 * category_distance)/float(w1+w2)
    indices = np.argsort(weighted_distance.ravel())[0:num_similar_items]
    news_list = pd.DataFrame({
        'headline': df['headline'][indices],
        'category': df['new_category'][indices]
    })
    result = news_list.to_dict('records')
    return result