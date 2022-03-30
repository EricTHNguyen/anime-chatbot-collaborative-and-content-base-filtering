from tensorflow.keras.models import load_model
import pandas as pd
anime= pd.read_csv(r"C:\Users\Erich\source\repos\Animebot\anime.csv")
rating= pd.read_csv(r"C:\Users\Erich\source\repos\Animebot\rating.csv")
model= load_model('model1.h5')
#create to make predictions
def make_pred(user_id, anime_id, model):
    return model.predict([np.array])
#created to make recomendation base on user_id and the model
def get_rec(user_id, model):
    user_id= int(user_id)
    user_ratings= rating[rating['user_id']== user_id]
    rec= rating[~rating['anime_id'].isin(user_rating['anime_id'])][['anime_id']].drop_duplicates()
    rec['rating_predict']= rec.apply(lambda x: make_pred(user_id, x['anime_id'], model), axis=1)
    final_rec= rec.sort_values(by='rating_predict', ascending=False).merge(anime[['anime_id','name','type','members']], on='anime_id').head(10)
    return final_rec.sort_values('rating_predict', ascending=False)[['name','type','rating_predict']]