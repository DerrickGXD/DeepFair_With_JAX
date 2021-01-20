"""
References for MovieLens 1M Dataset : 
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

References for Paper :
Jesús Bobadilla, Raúl Lara-Cabrera, Ángel González-Prieto, Fernando Ortega, (2020). 
DeepFair: Deep Learning for Improving Fairness in Recommender Systems[online]. 
Avaliable from :arXiv:2006.05255.
"""

import numpy as np
import nimfa
import scipy.sparse as sp
import pandas as pd
import warnings


warnings.filterwarnings("ignore")

user_data = pd.read_csv("ml-1m/users.dat",delimiter="::",header=None, names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
user_num = len(user_data)
print("Total number of users : {users}".format(users=user_num))

movie_data = pd.read_csv("ml-1m/movies.dat",delimiter="::",header=None, names=["MovieID", "Title", "Genres"])
movie_num = len(movie_data)
print("Total number of movies : {movies}".format(movies=movie_num))

rating_data = pd.read_csv("ml-1m/ratings.dat",delimiter="::",header=None, names=["UserID","MovieID","Rating","Timestamp"])
rating_num = len(rating_data)
print("Total number of ratings : {ratings}".format(ratings=rating_num))

mapping_movie = {} #map each unique MovieID with an integer
for count, movie in enumerate(list(movie_data['MovieID']), 0):
    mapping_movie[movie] = count

#Merge users, movies and ratings data by similar UserID and MovieID.
merged_data = pd.merge(user_data,rating_data,on="UserID",how="inner")
merged_data = pd.merge(merged_data,movie_data,on="MovieID",how="inner")
merged_data = merged_data.sort_values(by=['UserID'])
merged_data["UserID"] -= 1 #User ID ranges from [0,user_num-1] instead of [1,user_num]
merged_data["MovieID"] = merged_data["MovieID"].map(mapping_movie) #map each unique MovieID with an integer

rating_matrix = np.zeros((user_num,movie_num)) #Rating of each user (row) and item (column).
male_arr = np.zeros((user_num,)) #Every column represents user, 1 if male, 0 if female.
female_arr = np.zeros((user_num,)) #Every column represents user, 1 if female, 0 if male.

print("Populating rating matrix...")
#Populate rating_matrix, male_arr, female_arr.
for row in merged_data.itertuples(index=False):
    user_id = row.UserID
    movie_id = row.MovieID
    if(row.Gender=="F"):
        female_arr[user_id] = 1
    else:
        male_arr[user_id] = 1
    rating_matrix[user_id,movie_id] = row.Rating


#Reduce rating matrix dimension
pmf = nimfa.Pmf(rating_matrix, seed="random_vcol", rank=30, max_iter=5, rel_error=1e-5)
print("Reducing Matrix Dimension with PMF...")
pmf_fit = pmf()
print("Reducing Complete")

P = pmf_fit.basis()
Q = pmf_fit.coef().T
print("User matrix has shape of : {shape}".format(shape=P.shape))
print("Item matrix has shape of : {shape}".format(shape=Q.shape))

#Replace rating matrix with approximated matrix
rating_matrix = P @ Q.T

threshold = 3 #Indifference set (Refer to paper)
like_matrix = np.where(rating_matrix >= threshold, 1, 0) #Each row, column represents user and movie. Value is 1 if user likes the movie, otherwise 0. 
dislike_matrix = np.where((rating_matrix <= threshold) & (rating_matrix > 0), 1, 0) #Each row, column represents user and movie. Value is 1 if user dislikes the movie, otherwise 0. 

like_matrix_t = np.transpose(like_matrix)
dislike_matrix_t = np.transpose(dislike_matrix)

print("Calculating Item Minority Index")
#J_M is majority score for each item. 
#For this experiment, it is proportion of males who like the movie.
J_M = np.zeros((movie_num,))
for i in range(movie_num):
    U_up_i = like_matrix_t[i]
    U_down_i = dislike_matrix_t[i]
    U_M = male_arr
    total_votes_male = np.sum(U_up_i * U_M)+np.sum(U_down_i * U_M)
    total_likes_male = np.sum(U_up_i * U_M)
    if(total_votes_male!=0):
        J_M[i] = total_likes_male/total_votes_male
    else:
        J_M[i] = 0

#J_m is minority score for each item. 
#For this experiment, it is proportion of females who like the movie.
J_m = np.zeros((movie_num,))
for i in range(movie_num):
    U_up_i = like_matrix_t[i]
    U_down_i = dislike_matrix_t[i]
    U_m = female_arr
    total_votes_female = np.sum(U_up_i * U_m)+np.sum(U_down_i * U_m)
    total_likes_female = np.sum(U_up_i * U_m)
    if(total_votes_female!=0):
        J_m[i] = total_likes_female/total_votes_female
    else:
        J_m[i] = 0

#IM is Item Minority Index.
#For this experiment, to maintain index values in bounded range, we use preferences proportions as IM instead. 
IM = (J_M - J_m)

print("Calculating User Minority Index...")
#UM is User Minority Index.
UM = np.zeros((user_num,))
voted_matrix = np.where(rating_matrix != 0, 1, 0) #1 if voted, 0 if not voted.
above_threshold = rating_matrix - threshold*voted_matrix
for i in range(user_num):
    UM[i] = np.sum(np.squeeze(np.asarray(above_threshold[i])) * IM) / 5

#Normalise to [0,1] range
IM = (IM - IM.min())/(IM.max() - IM.min())
UM = (UM - UM.min())/(UM.max() - UM.min())

r_true = np.zeros((user_num,movie_num))
r_true = rating_matrix

beta_list = []
batch = 11
for i in range(batch):
    beta_list.append(i*0.1)

beta_loss = np.array(beta_list)
beta = np.array(beta_list).reshape(-1,1) #List of beta values to balance fairness and accuracy.

#Input data for neural network. For each rating, each vector is concatenation of user hidden factor, 
#item hidden factor and a beta value. 
experiment_data = np.zeros((rating_num*batch, 61)) 
#E_Accuracy required for loss function.
accuracy_data = np.zeros((rating_num*batch, 1))
#E_Fairness required for loss function.
fairness_data = np.zeros((rating_num*batch, 1))
#Beta required for loss function
beta_data = np.zeros((rating_num*batch, 1))

print("Storing Data for Experiment...")
index = 0
for row in merged_data.itertuples(index=False):
    user_id = row.UserID
    movie_id = row.MovieID
    p = np.squeeze(np.asarray(P[user_id])) #user_hidden_factor
    q = np.squeeze(np.asarray(Q[movie_id])) #item_hidden_factor
    pq = np.hstack((p,q))
    e_fairness = IM[movie_id] - UM[user_id]
    e_accuracy = r_true[user_id,movie_id]
    for i in range(batch):
        experiment_data[index,:] = np.hstack((pq,beta[i]))
        fairness_data[index,:] = np.array([e_fairness]) 
        beta_data[index,:] = np.array([beta[i]])
        accuracy_data[index,:] = np.array([e_accuracy])
        index += 1

print("Saving Data for Experiment...")
np.savez("experiment_data.npz", experiment_data=experiment_data, accuracy_data=accuracy_data, fairness_data=fairness_data, beta_data=beta_data)

print("Complete")
