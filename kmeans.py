import pandas as pd
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(r'u.user', sep='|', names=u_cols, encoding="latin-1")

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(r'u.data', sep='\t', names=r_cols, encoding="latin-1")

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv(r'u.item', sep='|', names=m_cols, usecols=range(5), encoding="latin-1")

users['sex'].replace(['F', 'M'], [0, 1], inplace=True)

users['occupation'] = users['occupation'].replace(['administrator'], [45000])
users['occupation'].replace(['artist'], [44000], inplace=True)
users['occupation'].replace(['doctor'], [169000], inplace=True)
users['occupation'].replace(['educator'], [64000], inplace=True)
users['occupation'].replace(['engineer'], [71000], inplace=True)
users['occupation'].replace(['entertainment'], [58000], inplace=True)
users['occupation'].replace(['executive'], [75000], inplace=True)
users['occupation'].replace(['healthcare'], [65000], inplace=True)
users['occupation'].replace(['homemaker'], [19000], inplace=True)
users['occupation'].replace(['lawyer'], [81000], inplace=True)
users['occupation'].replace(['librarian'], [49000], inplace=True)
users['occupation'].replace(['none'], [0], inplace=True)
users['occupation'].replace(['other'], [0], inplace=True)
users['occupation'].replace(['programmer'], [61000], inplace=True)
users['occupation'].replace(['retired'], [16000], inplace=True)
users['occupation'].replace(['salesman'], [30000], inplace=True)
users['occupation'].replace(['scientist'], [77000], inplace=True)
users['occupation'].replace(['student'], [4000], inplace=True)
users['occupation'].replace(['technician'], [42000], inplace=True)
users['occupation'].replace(['writer'], [57000], inplace=True)
users['occupation'].replace(['marketing'], [62000], inplace=True)

zip_strings = ['T8H1N','V3N4P','L9G2B','E2A4H','V0R2M','Y1A6B','V5A2B','M7A1A','M4J2K','R3T5K','T8H1N','N4T1A','V0R2H','K7L5J','V1G4L','L1V3W','N2L5N','E2E3R']
users = users[~users['zip_code'].isin(zip_strings)]

users['zip_code'] = pd.to_numeric(users['zip_code'])
normalized_users = users.copy()
for feature_name in users.columns:
    if feature_name != 'user_id':
        max_value = users[feature_name].max()
        min_value = users[feature_name].min()
        normalized_users[feature_name] = (users[feature_name] - min_value) / (max_value - min_value)

