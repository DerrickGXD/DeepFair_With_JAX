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

import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax
import optax 
from jax import jit, grad
import time
import warnings
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class MLN(hk.Module):
    
    def __init__(self):
        super().__init__()
        self.input_layer = hk.Linear(80)
        self.internal_1 = hk.Linear(10)
        self.internal_2 = hk.Linear(1)
        self.rng = jax.random.PRNGKey(42)     
    
    def __call__(self, x):
        x = self.input_layer(x)
        x = jax.nn.relu(x)
        x = self.internal_1(x)
        x = jax.nn.relu(x)
        x = hk.dropout(self.rng,0.2,x)
        x = self.internal_2(x)
        x = jax.nn.sigmoid(x)

        return x


#Split training and testing data to approximately 80% and 20%
batch = 11
training_num_rating = 800168
testing_num_rating = 200041

training_num_data = training_num_rating*batch
testing_num_data = testing_num_rating*batch

filename = "best_param"
outfile = open(filename,'rb')


print("Loading data ...")
data = np.load('experiment_data.npz')

experiment_data = data['experiment_data'][:testing_num_data]
accuracy_data = data['accuracy_data'][:testing_num_data]
fairness_data = data['fairness_data'][:testing_num_data]
beta_data = data['beta_data'][:testing_num_data]

# #Normalise accuracy and fairness to same range [0,1]
accuracy_data = (accuracy_data - accuracy_data.min())/(accuracy_data.max()-accuracy_data.min())
# fairness_data = (fairness_data - fairness_data.min())/(fairness_data.max()-fairness_data.min())

experiment_data_jnp = jnp.asarray(experiment_data, dtype=np.float32)
accuracy_data_jnp = jnp.asarray(accuracy_data, dtype=np.float32)
fairness_data_jnp = jnp.asarray(fairness_data, dtype=np.float32)
beta_data_jnp = jnp.asarray(beta_data, dtype=np.float32)

def MLN_fn(data):
    mln = MLN()
    return mln(data)


def loss_fn(params, input_data, accuracy, fairness, beta):
    out = model.apply(params, rng, input_data)
    e_accuracy = (out - accuracy)**2
    e_fairness = ((1-out) - fairness)**2 #If IM and UM difference is low, recommendation score should be high to minimize fairness
    acc_loss = jnp.mean(e_accuracy)
    fairness_loss = jnp.mean(e_fairness)
    return acc_loss, fairness_loss 

model = hk.transform(MLN_fn)
rng = jax.random.PRNGKey(42)
params = pickle.load(outfile)
previous_loss = float('inf')
print("Start Testing ...")

loss = 0
cur_time = time.time()

beta_list = []
acc_list = []
fairness_list = []

for i in range(batch):
    beta_value = i * 0.1
    input_data = experiment_data_jnp[i:testing_num_data:batch]
    accuracy = accuracy_data_jnp[i:testing_num_data:batch]
    fairness = fairness_data_jnp[i:testing_num_data:batch]
    beta = beta_data_jnp[i:testing_num_data:batch]
    acc_loss, fairness_loss = loss_fn(params,input_data,accuracy,fairness,beta)
    beta_list.append(float(beta_value))
    acc_list.append(float(acc_loss))
    fairness_list.append(float(fairness_loss))


beta_list = np.array(beta_list)
acc_list = np.array(acc_list)
fairness_list = np.array(fairness_list)
acc_list_norm = (acc_list - min(acc_list))/(max(acc_list)-min(acc_list))
fairness_list_norm = (fairness_list - min(fairness_list))/(max(fairness_list)-min(fairness_list))

print(fairness_list)

print("End Testing. Displaying results ...")

plt.plot(beta_list, acc_list_norm, 'k-', label='accuracy error')
plt.plot(beta_list, fairness_list_norm, 'b--', label='fairness error')
plt.xlabel("beta")
plt.ylabel("normalised error")
plt.legend(loc='upper right', fontsize='small')
plt.show()
