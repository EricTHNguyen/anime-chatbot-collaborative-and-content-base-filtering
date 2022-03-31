import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loads in the anime and rating csv 
animes= pd.read_csv(r"C:\Users\Erich\source\repos\Animebot\anime.csv")
ratings= pd.read_csv(r"C:\Users\Erich\source\repos\Animebot\rating.csv")
#print(animes.head())
#print("The shape of the data is (row, col):"+ str(animes.shape))
#print(animes.info())
#print(ratings.head())
#print("The shape of the data is (row, col):"+ str(ratings.shape))
#print(ratings.info())

#merging the rating and anime dataframe
main_dataset= pd.merge(animes,ratings, on="anime_id", suffixes=['','_user'])
#now we are gonna rename the columns
main_dataset= main_dataset.rename(columns={'name':'anime_title','rating_user':'user_rating'})
print(main_dataset.head())

#Creating dataframe for ratings
anime_rating= main_dataset.dropna(axis=0, subset=['anime_title'])
animerating_count=(main_dataset.groupby(by=['anime_title'])['user_rating'].count().reset_index().rename(columns={'raitng': 'totalRatingCount'})[['anime_title','user_rating']])
#replacing -1 with nana
anime=main_dataset.copy()
anime['user_rating'].replace({-1:np.nan}, inplace=True)
print(anime.head())
#dropping null values
anime= anime.dropna(axis=0,how='any')
anime.isnull().sum()
#filtering user_id
counts= anime['user_id'].value_counts()
anime=anime[anime['user_id'].isin(counts[counts>=200].index)]
#pviot the table where row is title and columns as user id
anime_pivot= anime.pivot_table(index='anime_title',columns='user_id', values='user_rating').fillna(0)
print(anime_pivot.head())
#collabarative filtering
#creating the sparse matrix
anime_matrix= csr_matrix(anime_pivot.values)
#fit the model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(anime_matrix)

#getting a random anime title and getting the recommendation for it
query_index= np.random.choice(anime_pivot.shape[0])
#print(query_index)
distances, indices= model_knn.kneighbors(anime_pivot.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
for i in range(0, len(distances.flatten())):
    if i==0:
        print( "Recommendations for {0}:\n".format(anime_pivot.index[query_index]))
    else:
        print('{0}:{1}, with distance of {2}:'.format(i, anime_pivot.index[indices.flatten()[i]], distances.flatten()[i]))