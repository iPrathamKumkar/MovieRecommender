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

# X = normalized_users
# distorsions = []
# for k in range(2, 20):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(X)
#     distorsions.append(kmeans.inertia_)

# fig = plt.figure(figsize=(15, 5))
# plt.plot(range(2, 20), distorsions)
# plt.grid(True)
# plt.title('Elbow curve')
# plt.show()

users_ratings = pd.merge(normalized_users, ratings)

u_train = normalized_users[normalized_users['user_id'] <= 848].drop(columns = 'user_id')
u_test = normalized_users[normalized_users['user_id'] > 848].drop(columns = 'user_id')

r_train = ratings[ratings['user_id'] <= 848]
r_test = ratings[ratings['user_id'] > 848]

kmeans = KMeans(n_clusters=7, random_state=0).fit(u_train)
# print('labels',labels)
labels = kmeans.labels_
age = int(input("Enter your age: "))
sex = input("Enter your sex (M/F): ")
if sex == 'M':
    sex = int(1)
else:
    sex = int(0)

def get_salary(occ):
    salary = {"administrator": 45000, "doctor": 169000, "artist": 44000, "educator": 64000, "engineer": 71000, "entertainment": 58000,
              "executive": 75000, "healthcare": 65000, "homemaker": 19000, "lawyer":81000, "librarian": 49000, "marketing": 62000,
              "none": 0, "other": 0, "programmer": 61000, "retired":16000,"salesman": 30000, "scientist": 77000, "student": 4000,
              "technician": 42000, "writer": 57000}
    return salary[occ]


occupation = input("Enter your occupation: ")
occupation = get_salary(occupation)
zip_code = int(input("Enter your zip code: "))

new_user = pd.DataFrame({"user_id":[users['user_id'].max()+2], "age":[(age-users['age'].min())/(users['age'].max()- users['age'].min())], "sex":[sex], "occupation":[(occupation-users['occupation'].min())/(users['occupation'].max()- users['occupation'].min())], "zip_code":[(zip_code-users['zip_code'].min())/(users['zip_code'].max()- users['zip_code'].min())]})

predicted_label = kmeans.predict(new_user.drop(columns = 'user_id'))
print(predicted_label)
print('centers', kmeans.cluster_centers_)

# for label in kmeans.labels_:
#     if predicted_label == label:

# mean = ratings.groupby(['user_id'], as_index=False, sort=False).mean().rename(columns={"rating":"mean_rating"})
# ratings = pd.merge(ratings, mean, on = "user_id", how = "left", sort="False")
# ratings['adjusted_ratings'] = ratings['rating']-ratings['mean_rating']

print('labels',kmeans.labels_)
cluster_users = []

cluster_ratings = ratings.drop(columns='unix_timestamp')
for i in range(len(kmeans.labels_)):
    if kmeans.labels_[i] in predicted_label:
        cluster_users.append(float(i+1))

print(cluster_users)

cluster_ratings = cluster_ratings[cluster_ratings['user_id'].isin(cluster_users)]
cluster_ratings = cluster_ratings.groupby('movie_id').filter(lambda x : len(x) > 10)
new_ratings = cluster_ratings.groupby('movie_id', as_index=False)['rating'].mean()
recommend = new_ratings.sort_values(by=['rating'], ascending=False).head(5)
#
recommend = pd.merge(recommend, movies)
recommend = recommend.drop(columns=['release_date','video_release_date','imdb_url'])
print(recommend.head(10))