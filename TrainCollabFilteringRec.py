from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Flatten, concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.utils import model_to_dot
from IPython.display import SVG

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

import warnings
warnings.filterwarnings('ignore')

anime= pd.read_csv(r"C:\Users\Erich\source\repos\Animebot\anime.csv")
rating= pd.read_csv(r"C:\Users\Erich\source\repos\Animebot\rating.csv")
#clean the data dropping null values
anime_null= anime.columns[anime.isna().any()]
anime[anime_null].isna().sum()
anime.dropna(inplace=True)
def cleantext(text):
    text=re.sub(r'&quot;','', text)
    text=re.sub(r'.hack//','', text)
    text=re.sub(r'&#039;','', text)
    text=re.sub(r'A&#039;s','', text)
    text=re.sub(r'I&#039;','', text)
    text=re.sub(r'&amp;','', text)
    return text
anime['name']= anime['name'].apply(cleantext)
##filter rating from 6 to 10
mask= (rating['rating']==-1)|(rating['rating']==1)|(rating['rating']==2)|(rating['rating']==3)|(rating['rating']==4)|(rating['rating']==5)
rating=rating.loc[~mask]
#change it from 6 to 10 to  1 to 5
def change_rating(rating):
    if rating==6:
        return 1
    elif rating==7:
        return 2
    elif rating==8:
        return 3
    elif rating==9:
        return 4
    elif rating==10:
        return 5
    rating['rating']=rating['rating'].apply(change_rating)

#filter user_id from 1 to 10000
rating= rating[rating['user_id']<10000]
#now use label encoder to target labels with value between 0 and n-1
user_en= LabelEncoder()
rating['user_id']= user_en.fit_transform(rating['user_id'])
anime_en=LabelEncoder()
rating['anime_id']= anime_en.fit_transform(rating['anime_id'])
#find unique total for user and anime
userid_nunique= rating['user_id'].nunique()
anime_nunique= rating['anime_id'].nunique()
#creating collab rec model
def Recommender(n_users, n_anime, n_dim):
    #users
    user= Input(shape=(1,))
    U=Embedding(n_users, n_dim)(user)
    U=Flatten()(U)

    #Anime
    anime= Input(shape=(1,))
    A= Embedding(n_anime, n_dim)(anime)
    A= Flatten()(A)
    
    #Merged user and anime. Create model
    merged_vector= concatenate([U,A])
    dense_1= Dense(128, activation='relu')(merged_vector)
    dropout= Dropout(0.5)(dense_1)
    final=Dense(1)(dropout)
    model= Model(inputs=[user, anime], outputs=final)
    model.compile(optimizer=Adam(0.001),loss='mean_squared_error')
    return model
#now use the model on the unique user and anime
model= Recommender(userid_nunique,anime_nunique, 100)
SVG(model_to_dot(model, show_shapes=True, show_layer_names=True))
model.summary()
# train and test data on the model
X= rating.drop(['rating'], axis=1)
y=rating['rating']
X_train, X_val, y_train, y_val= train_test_split(X,y, test_size=.1, stratify=y, random_state=2020)
check= ModelCheckpoint('Anime_rec.h5', monitor='val_loss', verbose=0, save_best_only=True)
history= model.fit(x=[X_train['user_id'], X_train['anime_id']],
                   y=y_train,
                   batch_size=64,
                   epochs=999,
                   verbose=1,
                   validation_data=([X_val['user_id'], X_val['anime_id']], y_val),
                   callbacks=[check])

print("model is created")