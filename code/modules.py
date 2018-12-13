
import numpy as np

# modules.py
# training layers, components

def pca(X, new_dims):
    # code from cs231n notes
    # faster versions are implenmented in lm_models.py

    # Assume input data matrix X of size [N x D]
    X -= np.mean(X, axis = 0) # zero-center the data (important)
    cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
    U,S,V = np.linalg.svd(cov)
    Xrot = np.dot(X, U) # decorrelate the data
    Xrot_reduced = np.dot(X, U[:,:new_dims]) # Xrot_reduced becomes [N x new_dims]

    return Xrot_reduced

def whiten(X):
    # faster versions are implenmented in lm_models.py

    # Assume input data matrix X of size [N x D]
    X -= np.mean(X, axis = 0) # zero-center the data (important)
    cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
    U,S,V = np.linalg.svd(cov)
    Xrot = np.dot(X, U) # decorrelate the data

    # whiten the data:
    # divide by the eigenvalues (which are square roots of the singular values)
    Xwhite = Xrot / np.sqrt(S + 1e-5)

    return Xwhite

def run_test():
    print('test modules')

if __name__ == "__main__":
    run_test()

