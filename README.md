# DeepFair With JAX
Reimplementation of Deep Learning Model via Multilayer Neural Network Training (MLN) in [DeepFair: Deep Learning for Improving Fairness in Recommender Systems](https://arxiv.org/pdf/2006.05255v1.pdf) with [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku). Uses **just-in-time (JIT)** compiling to speed up training and evaluation process. There are some difference between the original idea in the paper and the implementation in this code, hence results can be different from what is shown in the paper.  Feel free to assign an issue if there is any suggestions or corrections to improve the code. 

# Dataset
[MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/). Same as the dataset used for experiment in DeepFair's research paper.

# Creating NumPy dataset
```
python test.py
```
This script will generate `experiment_data.npz` which contains input vectors and other vectors for calculating loss function.

# Data Visualisation
```
python dist.py
```
This script will generate proportions of category of users in MovieLens (top-left), Item Minority Index (IM) and User Minority Index (UM) distribution (top-right), UM comparative between groups (bottom-left) and accuracy of user classification via UM (bottom-right).

# Training Data
```
python train.py
```
This script will use `experiment_data.npz` to train the model. The parameters for training, e.g batch, learning rate, have not been optimised yet. After complete training, the parameters for model will be saved in `best_param.pkl` file.

# Testing Data
```
python test.py
```
This script will use `experiment_data.npz` to test the model and plot the normalised error of accuracy and fairness for each hyperparameter (beta).

# Future Plans
Optimising the training in GPU. Implement the model in other frameworks and compare **JIT** performance between frameworks. Improvised the modified calculation of loss function.

# References
1. Jesús Bobadilla, Raúl Lara-Cabrera, Ángel González-Prieto, Fernando Ortega (2020). DeepFair: Deep Learning for Improving Fairness in Recommender Systems[online]. Avaliable from :arXiv:2006.05255.
2. F. Maxwell Harper and Joseph A. Konstan (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872
