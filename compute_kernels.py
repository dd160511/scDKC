import numpy as np
import sklearn
import torch
from sklearn.metrics import pairwise_distances, pairwise_kernels


def gaussian(sqdist, sigma=1):
    K = np.exp(-sqdist / (2 * (sigma * sqdist.max()) ** 2))
    tmp = K.max()
    if tmp == 0:
        tmp = 1
    return K/tmp

# def gaussian(Pseq, sigma=1):
#     return torch.exp(-Pseq / sigma ** 2)

def polynomial(Pdot, c=0, d=1):
    return (Pdot + c) ** d

def cosine(Pdot):
    return Pdot / torch.outer(torch.diag(Pdot), torch.diag(Pdot)) ** 0.5
def tanh(Pdot):
    return torch.tanh(Pdot)


def compute_kernelsm(X):
    #Pseq = torch.pairwise_distance(X, X)
    #Pseq = pairwise_distances(X.detach().numpy() , X.detach().numpy() , metric='sqeuclidean', n_jobs=8)
    #Pseq=pairwise_kernels(X.detach().numpy().reshape(X.shape[0],-1),metric='rbf' ,gamma=2,n_jobs=8)
    Pdot = torch.mm(X, X.T)
    K = torch.zeros((12, X.shape[0], X.shape[0]))
    list_sigmas = [1e-2, 5e-2, 1e-1, 1, 10, 50, 100]
    for i in range(len(list_sigmas)):
        K[i] = gaussian(Pdot, sigma=list_sigmas[i])

    list_c = [0, 1]
    list_d = [2, 4]
    for i in range(len(list_c)):
        for j in range(len(list_d)):
            K[7 + i * 2 + j] = polynomial(Pdot, c=list_c[i], d=list_d[i])
    K[11] = cosine(Pdot)
    return K
def compute_kernels(X):

    Pdot = torch.mm(X, X.T)

    K = cosine(Pdot)
    return K

def compute_kernelst(X):

    Pdot = torch.mm(X, X.T)

    K = tanh(Pdot)
    return K

def compute_kernelsp(X):

    Pdot = torch.mm(X, X.T)

    K = polynomial(Pdot)
    return K
def compute_kernelsp1(X):

    Pdot = torch.mm(X, X.T)

    K = polynomial(Pdot, c=1, d=2)
    return K
# def compute_kernelsg(X):
#     Pseq =pairwise_distances(X, X, metric='euclidean', n_jobs=8)
#     K = gaussian(Pseq, sigma=2)
#
#     return K
#
# def compute_kernelsg1(X):
#     Pseq = torch.mm(X, X.T)#pairwise_distances(X, X, metric='sqeuclidean', n_jobs=-1)
#     K = gaussian(Pseq, sigma=0.1)
#
#     return K
# def compute_kernelsg2(X):
#     Pseq = torch.mm(X, X.T)#pairwise_distances(X, X, metric='sqeuclidean', n_jobs=-1)
#     K = gaussian(Pseq, sigma=10)
#
#     return K
# def compute_kernelsg3(X):
#     Pseq = torch.mm(X, X.T)#pairwise_distances(X, X, metric='sqeuclidean', n_jobs=-1)
#     K = gaussian(Pseq, sigma=2)

  #  return K
# def compute_kernelsg(data, gamma = 20):
#     gamma = gamma / data.shape[1]
#     # out = -torch.cdist(data, data, p = 2)
#     K = torch.cdist(data, data, p = 2) ** 2 * (-gamma)
#     out = torch.exp(K)
#     return out

def compute_kernelsg(data, γ=0.5):
	K = sklearn.metrics.pairwise.rbf_kernel(data, gamma=γ)
	return K
def compute_kernelsg1(data, γ=0.1):
	K = sklearn.metrics.pairwise.rbf_kernel(data, gamma=γ)
	return K
def compute_kernelsg2(data, γ=1):
	K = sklearn.metrics.pairwise.rbf_kernel(data, gamma=γ)
	return K
def compute_kernelsg3(data, γ=0.01):
	K = sklearn.metrics.pairwise.rbf_kernel(data, gamma=γ)
	return K