import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
#loads in the anime and rating csv 
animes= pd.read_csv(r"C:\Users\Erich\source\repos\Animebot\anime.csv")
ratings= pd.read_csv(r"C:\Users\Erich\source\repos\Animebot\rating.csv")
indices= pd.Series(animes.index, index=animes['name'])
#create content rec, using cosine similarity.
def genre_rec(title, highest_rating=False,similarity=False):
    #if it is not a high rating and or similar and not
   if highest_rating== False:
    if similarity== False:
        idx= indices[title]
        sim= list(enumerate(cosine_sim[idx]))
        sim= sorted(sim, key=lambda x: x[1], reverse=True)
        sim=sim[1:11]
        aniindices= [i[0] for i in sim]
        return pd.DataFrame({'Anime':animes['name'].iloc[aniindices].values, 'Type':animes['type'].iloc[aniindices].values})
    elif similarity== True:
        idx=indices[title]
        sim= list(enumerate(cosine_sim[idx]))
        sim=sorted(sim, key=lambda x: x[1], reverse=True)
        sim= sim[1:11]
        aniindices=[i[0]for i in sim]
        similarity=[i[1]for i in sim]
        return pd.DataFrame({'Anime':animes['name'].iloc[aniindices].values, 'Type':animes['type'].iloc[aniindices].values})
     #if it is a high rating and or similar and not
    elif highest_rating==True:
     if similarity== False:
          idx=indices[title]
          sim= list(enumerate(cosine_sim[idx]))
          sim=sorted(sim, key=lambda x: x[1], reverse=True)
          sim= sim[1:11]
          aniindices=[i[0]for i in sim]
          result_df= pd.DataFrame({'Anime':animes['name'].iloc[aniindices].values, 'Type':animes['type'].iloc[aniindices].values, 'Rating':animes['rating'].iloc[aniindices].values})
          return result_df.sort_values('Rating', ascending=False)
    elif similarity== True:
        idx=indices[title]
        sim= list(enumerate(cosine_sim[idx]))
        sim=sorted(sim, key=lambda x: x[1], reverse=True)
        sim= sim[1:11]
        aniindices=[i[0]for i in sim]
        similarity=[i[1]for i in sim]
        result_df= pd.DataFrame({'Anime':animes['name'].iloc[aniindices].values, 'Type':animes['type'].iloc[aniindices].values, 'Rating':animes['rating'].iloc[aniindices].values})
        return result_df.sort_values('Rating', ascending=False)