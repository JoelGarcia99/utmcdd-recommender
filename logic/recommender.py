import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # used for data transformation
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # used for data transformation
from sklearn.metrics.pairwise import cosine_similarity

from models.anime import Anime

'''
The dataset used here can be found online via kaggle repository at
https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database 

Here's a quick documentation of the columns

anime_id - myanimelist.net's unique id identifying an anime.
name - full name of anime.
genre - comma separated list of genres for this anime.
type - movie, TV, OVA, etc.
episodes - how many episodes in this show. (1 if movie).
rating - average rating out of 10 for this anime.
members - number of community members that are in this anime's "group".
'''

class Recommender:
    def __init__(self):
        self.init_data()
        self.data_transformation()
        self.build_similarity_matrix()

        # a hashmap where the key is the id and the value is a reference to the Anime
        self.data_hashmap = {
            record['anime_id']: Anime(record) for record in self.original_data.to_dict('records')
        }

    def init_data(self):
        self.original_data = pd.read_csv("./dataset/anime.csv")
        self.data = self.original_data

    def data_transformation(self):
        # handling non numerical values
        self.data['members'] = pd.to_numeric(self.data['members'], errors='coerce')
        self.data['episodes'] = pd.to_numeric(self.data['episodes'], errors='coerce')
        self.data['rating'] = pd.to_numeric(self.data['rating'], errors='coerce')

        # Handle missing values (NaN) by filling with the mean, or drop rows
        self.data.fillna({
            'members': self.data['members'].mean(),
            'episodes': self.data['episodes'].mean(),
            'rating': self.data['rating'].mean()
        }, inplace=True)

        #================= ENCODING
        # getting categories as dummies (0s and 1s columns) so we avoid biases introduced
        # by this column if we only transform it into numerical, if we do so then categories
        # are gonna be labelled from 0 to N where N > 0, of course and that's not the meaning
        # of these columns
        genres = self.data['genre'].str.get_dummies(sep=',')
        self.data = pd.concat([self.data, genres], axis=1)

        # one hot encoding for other columns
        type_encoder = OneHotEncoder()
        type_encoded = type_encoder.fit_transform(self.data[['type']])
        type_encoded_data = pd.DataFrame(type_encoded.toarray(), columns=type_encoder.get_feature_names_out())
        self.data = pd.concat([self.data, type_encoded_data], axis=1)


        # standardizing numerical columns to avoid biases
        scaler = MinMaxScaler()
        self.data[['members', 'episodes', 'rating']] = scaler.fit_transform(self.data[['members', 'episodes', 'rating']])

        # Create TF-IDF vectorization for the 'name' column
        vectorizer = TfidfVectorizer()
        name_tfidf_data = vectorizer.fit_transform(self.data['name'])
        name_tfidf_data = pd.DataFrame(name_tfidf_data.toarray(), columns=vectorizer.get_feature_names_out())
        self.data = pd.concat([self.data, name_tfidf_data], axis=1)

        # droping unused columns because we already transformed them and we don't need them anymore
        self.data = self.data.drop(columns=['name', 'genre', 'type'])

        # droping anime_id column because it's not useful
        self.data = self.data.drop(columns=['anime_id'])

    '''
    Builds the similarity matrix based on cosine
    '''
    def build_similarity_matrix(self):
        # ==================== matrix creation
        feature_matrix = self.data.to_numpy()

        # calculating cosine similarity
        self.__cosine_sim = cosine_similarity(feature_matrix)

    def recommend(self, anime_id, top_n=5):
        index = self.data.index[self.original_data['anime_id'] == anime_id].tolist()[0]

        # get similarity scores for the given anime against all other animes
        similarity_scores = list(enumerate(self.__cosine_sim[index]))

        # sort the similarity scores in descending order so we get the top N nearest
        # neighbors first
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # get the indices of the top N nearest neighbors
        top_indices = similarity_scores[1:top_n+1]

        # get the names of the top N nearest neighbors
        top_names = [(self.original_data['anime_id'].iloc[i], score) for i, score in top_indices]

        return top_names
