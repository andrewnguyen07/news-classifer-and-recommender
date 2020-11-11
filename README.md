# News Classifier & Recommender

This project shows 2 Jupyter Notebooks: (1) that builds a neural network model (Bidirectional LSTM) classifying the news into its correct category and (2) that leverages the classification and subsequently recommends a similar list of news. The project used Kaggle's [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset), so feel free to explore my work that follows as well as other notebooks from Kaggle.
The repository is also served as a hub of different deployment techniques using FastAPI and Unicorn: (1) manual input and (2) upload file

* news_classifier.ipynb: for a larger volume of text needed to train the model, the notebook combined "headline" and "description" data to build the vocabulary. Then I used TextVectorization (tensorflow.keras) to create a vectorizer that helps process the text input and acts as the first layer in the neural network model. Word Embedding is also leveraged as the 2nd layer to pad the sequences of data to those of equal length for Bidrectional LSTM to be able to learn. The model achieved quite a low accuracy of 75%, which can be optimized given more text data is available for use.
* news-recommender.ipynb: this notebook establishes the connection with the classifider model which defines functions that calculates the euclidean similarity distance between the prediction text and the existing database. TfidfVectorizer is used in these functions as a baseline for computation.
* production_html.py: this file acts as a deployment document using Flask with HTML templates to input the news on the server and make predictions. 

## Requirements

* pandas
* numpy
* matplotlib
* seaborn
* sklearn
* tensorflow
* flask
