from flask import Flask, request, jsonify, render_template # loading in Flask
import pandas as pd # loading pandas for reading csv
import tensorflow as tf 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
import numpy as np 
from news_classifier import inverse_prediction
from news_classifier import headline_desc_mapping
from news_classifier import text_preprocessing
from news_classifier import headline_and_desc_tfidf_recommender
from news_classifier import text_and_category_tfidf_recommender
from news_classifier import df

# creating a Flask application
app = Flask(__name__, template_folder="templates")

# Load the model
model = tf.keras.models.load_model('/Users/andrewnguyen/gitfolder/nlp2/news_classifier')

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = request.form.get('headline', 'description')
        df_new = pd.DataFrame({
            'headline': data[0],
            'description': data[1]
        }, index=[0])
        text = headline_desc_mapping(df_new['headline'], df_new['description'])
        text = pd.DataFrame(text, columns=['news'])
        text = text_preprocessing(text['news'])
        pred = model.predict(text)
        pred_label = inverse_prediction(pred)
        news_list = text_and_category_tfidf_recommender(
            df, df['headline_and_desc'], text.values, 
            df['new_category'], pred_label, 
            num_similar_items=10, w1=0.2, w2=0.8)
        column_names = ['headline', 'category']
        return render_template('index.html', 
        result=str(pred_label), 
        lists=news_list, colnames=column_names)
    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(port=3000, debug=True)