import pandas as pd
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
#
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
users['occupation'].replace([' librarian'], [62000], inplace=True)
users['occupation'].replace(['none'], [0], inplace=True)
users['occupation'].replace(['other'], [0], inplace=True)
users['occupation'].replace(['random programmer'], [61000], inplace=True)
users['occupation'].replace(['retired'], [16000], inplace=True)
users['occupation'].replace(['salesman'], [30000], inplace=True)
users['occupation'].replace(['scientist'], [77000], inplace=True)
users['occupation'].replace(['student'], [4000], inplace=True)
users['occupation'].replace(['technician'], [42000], inplace=True)
users['occupation'].replace(['writer'], [57000], inplace=True)

