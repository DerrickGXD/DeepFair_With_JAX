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
import matplotlib.pyplot as plt

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

threshold = 3 #Indifference set (Refer to paper)
like_matrix = np.where(rating_matrix >= threshold, 1, 0) #Each row, column represents user and movie. Value is 1 if user likes the movie, otherwise 0. 
dislike_matrix = np.where((rating_matrix <= threshold) & (rating_matrix > 0), 1, 0) #Each row, column represents user and movie. Value is 1 if user dislikes the movie, otherwise 0. 

like_matrix_t = np.transpose(like_matrix)
dislike_matrix_t = np.transpose(dislike_matrix)

print("Calculating Item Minority Index ...")

#J_M is majority score for each item. 
Vote_M = np.zeros((movie_num,))
J_M = np.zeros((movie_num,))
for i in range(movie_num):
    U_up_i = like_matrix_t[i]
    U_down_i = dislike_matrix_t[i]
    U_M = male_arr
    total_votes_male = np.sum(U_up_i * U_M)+np.sum(U_down_i * U_M)
    total_likes_male = np.sum(U_up_i * U_M)
    total_dislikes_male = np.sum(U_down_i * U_M)
    Vote_M[i] = total_votes_male
    if(total_votes_male!=0):
        J_M[i] = (total_likes_male-total_dislikes_male)/total_votes_male
    else:
        J_M[i] = 0

#J_m is minority score for each item. 
Vote_F = np.zeros((movie_num,))
J_m = np.zeros((movie_num,))
for i in range(movie_num):
    U_up_i = like_matrix_t[i]
    U_down_i = dislike_matrix_t[i]
    U_m = female_arr
    total_votes_female = np.sum(U_up_i * U_m)+np.sum(U_down_i * U_m)
    total_likes_female = np.sum(U_up_i * U_m)
    total_dislikes_female = np.sum(U_down_i * U_m)
    Vote_F[i] = total_votes_female
    if(total_votes_female!=0):
        J_m[i] = (total_likes_female-total_dislikes_female)/total_votes_female
    else:
        J_m[i] = 0

#IM is Item Minority Index.
IM = (J_M - J_m)

#IM is [-2,2], normalise to [-1,1]
IM = (IM - IM.min())/(IM.max()-IM.min())
IM = (2*IM)-1


print("Calculating User Minority Index...")
#UM is User Minority Index.
UM = np.zeros((user_num,))
voted_matrix = np.where(rating_matrix != 0, 1, 0) #1 if voted, 0 if not voted.
above_threshold = rating_matrix - threshold
for i in range(user_num):
    movies_rating = above_threshold[i]
    voted_movies = voted_matrix[i]
    normalise_factor = (5-threshold) * np.sum(voted_movies)
    UM[i] = np.sum(movies_rating*IM*voted_movies)/normalise_factor



IM_new = np.around(IM, decimals=2)
UM_new = np.around(UM, decimals=2)


(IM_val, IM_freq) = np.unique(IM_new, return_counts=True)
(UM_val, UM_freq) = np.unique(UM_new, return_counts=True)


UM_val = np.insert(UM_val,0,-1)
UM_val = np.append(UM_val,1)
UM_freq = np.insert(UM_freq,0,0)
UM_freq = np.append(UM_freq,0)


#normalise IM_freq and UM_freq
IM_freq = IM_freq/IM_freq.max()
UM_freq = UM_freq/UM_freq.max()

#Get UM in male and female category
UM_male = UM * male_arr
UM_male = UM_male[UM_male!=0]
UM_female = UM * female_arr
UM_female = UM_female[UM_female!=0]

UM_male_new = np.around(UM_male, decimals=2)
UM_female_new = np.around(UM_female, decimals=2)


(UM_male_val, UM_male_freq) = np.unique(UM_male_new, return_counts=True)
(UM_female_val, UM_female_freq) = np.unique(UM_female_new, return_counts=True)
UM_male_freq = UM_male_freq/UM_male_freq.max()
UM_female_freq = UM_female_freq/UM_female_freq.max()


UM_male_val = np.insert(UM_male_val,0,-1)
UM_male_val = np.append(UM_male_val,1)
UM_male_freq = np.insert(UM_male_freq,0,0)
UM_male_freq = np.append(UM_male_freq,0)

UM_female_val = np.insert(UM_female_val,0,-1)
UM_female_val = np.append(UM_female_val,1)
UM_female_freq = np.insert(UM_female_freq,0,0)
UM_female_freq = np.append(UM_female_freq,0)

#Calculate users classications accuracy based on UM
UM_male_accurate = np.where(UM_male > 0, 1, 0) #1 if correctly classified, else 0
UM_male_correct = np.count_nonzero(UM_male_accurate)
UM_male_incorrect = UM_male.shape[0] - UM_male_correct
UM_male_accuracy = UM_male_correct*100/UM_male.shape[0]
UM_male_accuracy = round(UM_male_accuracy, 2)

UM_female_accurate = np.where(UM_female < 0, 1, 0) #1 if correctly classified, else 0
UM_female_correct = np.count_nonzero(UM_female_accurate)
UM_female_incorrect = UM_female.shape[0] - UM_female_correct
UM_female_accuracy = UM_female_correct*100/UM_female.shape[0]
UM_female_accuracy = round(UM_female_accuracy, 2)

UM_accuracy_table = [
    ["Category", "Correct", "Incorrect", "Correct%"],
    ["Female", UM_female_correct, UM_female_incorrect, UM_female_accuracy],
    ["Male", UM_male_correct, UM_male_incorrect, UM_male_accuracy]
]

plt.subplots_adjust(left=0.125, right=0.9, bottom=0.15, top=0.9, wspace=0.5, hspace=0.5)
plt.subplot(221)
plt.bar(["Male","Female"], [np.sum(male_arr),np.sum(female_arr)])
plt.title("Number of users")
plt.subplot(222)
plt.plot(IM_val, IM_freq, 'g-', label='items')
plt.plot(UM_val, UM_freq, 'k--', label='users')
plt.xlabel("minority value")
plt.ylabel("scaled number of items and number of users", fontsize='small')
plt.title("IM and UM distributions")
plt.legend(loc='upper right', fontsize='small')
plt.subplot(223)
plt.plot(UM_male_val, UM_male_freq, 'b--', label='male')
plt.plot(UM_female_val, UM_female_freq, 'r-', label='female')
plt.xlabel("minority value")
plt.ylabel("scaled number of items and number of users", fontsize='small')
plt.title("UM Comparative")
plt.legend(loc='upper right', fontsize='small')
plt.subplot(224)
table = plt.table(cellText=UM_accuracy_table, loc='center', fontsize='medium')
table.scale(1, 3)
table.auto_set_font_size(False)
table.set_fontsize('medium')
plt.title("User Classification via UM")
plt.axis('off')

plt.show()


