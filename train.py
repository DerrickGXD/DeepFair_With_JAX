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
from math import ceil
import warnings
import pickle

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
        x = self.internal_1(x)
        x = jax.nn.relu(x)
        x = hk.dropout(self.rng,0.2,x)
        x = self.internal_2(x)
        x = jax.nn.relu(x)
        
        return x

def split_shuffle_data(arr):
    np.random.shuffle(arr)
    data_len = arr.shape[0]
    vector_len = arr.shape[1]

    return arr[:,:vector_len-3], arr[:,-3].reshape(data_len,1), arr[:,-2].reshape(data_len,1), arr[:,-1].reshape(data_len,1)

#Split training and testing data to approximately 80% and 20%
batch = 11
training_num_rating = 800168
testing_num_rating = 200041

training_num_data = training_num_rating*batch
testing_num_data = testing_num_rating*batch

filename = "best_param"
outfile = open(filename,'wb')

n_epochs = 10
lr = 1e-4

print("Loading data ...")
data = np.load('experiment_data.npz')

experiment_data = data['experiment_data'][:training_num_data]
accuracy_data = data['accuracy_data'][:training_num_data]
fairness_data = data['fairness_data'][:training_num_data]
beta_data = data['beta_data'][:training_num_data]

print("Shuffling data ...")
concat_array = np.concatenate((experiment_data,accuracy_data,fairness_data,beta_data), axis=1)
experiment_data, accuracy_data, fairness_data, beta_data = split_shuffle_data(concat_array)

experiment_data_jnp = jnp.asarray(experiment_data, dtype=np.float32)


accuracy_data_jnp = jnp.asarray(accuracy_data, dtype=np.float32)
fairness_data_jnp = jnp.asarray(fairness_data, dtype=np.float32)
beta_data_jnp = jnp.asarray(beta_data, dtype=np.float32)

training_batch_size = 4*batch
num_batch = int(training_num_data/training_batch_size)

def MLN_fn(data):
    mln = MLN()
    return mln(data)

@jit
def loss_fn(params, input_data, accuracy, fairness, beta):
    out = model.apply(params, rng, input_data)
    e_accuracy = (out - accuracy)**2
    e_fairness = fairness
    loss = beta*e_accuracy + (1-beta)*e_fairness
    final_loss = jnp.mean(loss)
    return final_loss

@jit
def update(grads, opt_state, params):
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params

model = hk.transform(MLN_fn)
rng = jax.random.PRNGKey(42)
params = model.init(rng, experiment_data_jnp)
opt = optax.rmsprop(lr)
opt_state = opt.init(params)



print("Start Training ...")
for epoch in range(n_epochs):
    loss = 0
    cur_time = time.time()

    for i in range(num_batch):
        start = i * training_batch_size
        end = start + training_batch_size

        input_data = experiment_data_jnp[start:end,:]
        accuracy = accuracy_data_jnp[start:end,:]
        fairness = fairness_data_jnp[start:end,:]
        beta = beta_data_jnp[start:end,:]

        grads = grad(loss_fn)(params,input_data,accuracy,fairness,beta)
        params = update(grads, opt_state, params)

        loss += loss_fn(params,input_data,accuracy,fairness,beta)


    average_loss = loss/ceil(training_num_data/training_batch)
    print("Epoch {epoch} : {loss}".format(epoch=epoch,loss=average_loss))
    print("Time taken", time.time()-cur_time)

print("End Training. Saving best params...")

pickle.dump(params,outfile)
