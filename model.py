import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import  train_test_split
import pickle

import nltk
nltk.download('omw-1.4')

sentiment_model_file = './models/sentiment_analysis_rf_model.pkl'
recomm_model_file = './models/user_based_cf_recomm.pkl'
tfidf_vect_file = './models/tf_idf_vect.pkl'
product_sentiment = './models/product_sentiments.pkl'

class RecommendationModel():

    # preprocess the dataframe
    def preprocess_recomm_data(self, df):
        df['user_id'] = pd.factorize(df['reviews_username'])[0]
        df.rename(columns={'id':'product_id', 'reviews_rating':'rating', 'reviews_username':'user_name'}, inplace=True)
        return df

    # get sentiment score based on the reviews of the product
    def get_sentiment_score(self, recomm_df, product_name):
        prod_senti_model = pickle.load(open(product_sentiment, 'rb'))
        sentiment_score = prod_senti_model.loc[product_name].values[0]
        return sentiment_score

    # get 20 recommendations of product for a user - without sentiment analysis
    def get_product_recommendations(self, recomm_df, user_name, num_recommendations):
        usrid = recomm_df[recomm_df['user_name']==user_name]['user_id'].unique()
        recommendations = pickle.load(open(recomm_model_file, 'rb'))
        d = recommendations.loc[int(usrid)].sort_values(ascending=False)[0:num_recommendations]
        recommended_product = pd.DataFrame(d)
        recommended_product.reset_index(inplace=True)
        recommended_product['product_name'] = [recomm_df[recomm_df['product_id']==x]['name'].unique()[0] for x in recommended_product['product_id']]
        return recommended_product

    # get 5 best product recommendations based on the sentiments
    def get_top_products_based_on_sentiment(self, recomm_df, recommendations):
        recommendations['sentiment_score'] = [self.get_sentiment_score(recomm_df, x) for x in recommendations['product_name']]
        recommendations.sort_values(by='sentiment_score', ascending=False, inplace=True)
        top_recomm_products = recommendations['product_name'][0:5]
        return top_recomm_products

    # get recommendations
    def recommend(self, name):
        df = pd.read_csv("./data/sample30.csv")
        recom_df = self.preprocess_recomm_data(df)
        recommendations = self.get_product_recommendations(recom_df, name, 20)
        top_products = self.get_top_products_based_on_sentiment(recom_df, recommendations)
        return list(top_products)
    

    def __init__(self) -> None:
        pass

    
