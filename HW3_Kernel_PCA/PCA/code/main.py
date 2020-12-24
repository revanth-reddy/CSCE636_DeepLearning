import torch
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os


def test_pca(A, p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    Acentered = A  - A.mean(axis=1, keepdims=True)
    error = frobeniu_norm_error(Acentered, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(A, p):
    model = AE(d_hidden_rep=p)
    model.train(A, A, 128, 300)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w

def test_tune_ae(A, p, bs, ep):
    model = AE(d_hidden_rep=p)
    model.train(A, A, bs, ep)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w, error

if __name__ == '__main__':
    dataloc = "../data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)
#    ps = [50, 100, 150]
    ps = [32, 64, 128]
    ps = [64]
    err = []
    Ierr = []
    for p in ps:
        G = test_pca(A, p)
        final_w = test_ae(A, p)
        err.append(frobeniu_norm_error(G, final_w))
        Ierr.append(frobeniu_norm_error(G.T.dot(G), final_w.T.dot(final_w)))

    epochs = [2500]
    batch_sizes = [64]
  
    for p in ps:
      for bs in batch_sizes:
        for ep in epochs:
          final_w, error = test_tune_ae(A, p, bs, ep)
          err.append((bs,ep,error))

    print(err)
    print(Ierr)
    ### END YOUR CODE 
