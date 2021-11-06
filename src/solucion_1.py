import numpy as np
from numpy.lib import diff
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.io import loadmat

def oneVsAll(X, y, num_labels, lamb):
    """
    Entrenamiento de varios clasificadores por regresión logística
    """
    initial_theta = np.zeros(X.shape[1])
    all_theta = np.zeros((num_labels, X.shape[1]))
    y.shape = (y.shape[0],)

    for c in np.arange(1, num_labels + 1):
        result = scipy.optimize.fmin_tnc(fun_J, initial_theta, new_theta, args=(X, (y == c)*1, lamb))
        all_theta[c-1] = result[0]

    return all_theta

def sigmoide_fun(Z):
    G = 1 / (1 + (np.exp(-Z)))

    return G

def fun_J(Theta, X, Y, lamb):
    """
        Calculates the J function of the cost
        of the Logistic Regresion    
    """
    m = X.shape[1]
    S = sigmoide_fun(np.dot(X, Theta))
    Sum1 = np.dot(Y, np.log(S))

    # This add is to dodge the log(0)
    Diff = (1 - S) + 0.00001
    Sum2 = np.dot((1 - Y), np.log(Diff))
    # First part
    Sum = Sum1 + Sum2
    Sum = (-1 / m) * Sum
    # Lambda part
    Sum3 = np.sum(np.power(Theta, 2))
    Sum += (lamb / (2 * m)) * Sum3

    return Sum 

def new_theta(Theta, X, Y, lamb):
    """
        Calculate the new value of Theta with matrix
    """
    m = X.shape[1]
    Z = np.matmul(X, Theta)
    S = sigmoide_fun(Z)
    Diff = S - Y

    X_t = np.transpose(X)
    NewTheta = (1 / m) * np.matmul(X_t, Diff) + (lamb/m) * Theta
    NewTheta[0] -= (lamb/m) * Theta[0]

    return NewTheta

def evaluation(all_thetas, X, y): 
    total = X.shape[0]
    percentage = 0
    maxH = -9999
    maxIndex = -1
    for i in range(total):
        for j in range(all_thetas.shape[0]):
            sig = sigmoide_fun(np.dot(X[i], all_thetas[j]))
            if  sig > maxH:
                maxH = sig
                maxIndex = j
        if maxIndex == y[i]:
            percentage += 1

    return (percentage / total) * 100

#main
data = loadmat ('./src/ex3data1.mat')
y = data ['y']
X = data ['X']

# Number of images
m = X.shape[0]
XX = np.hstack([np.ones([m, 1]), X])

all_thetas = oneVsAll(XX, y, 10, 0.1)
print("PERCENTAGE: ", evaluation(all_thetas, XX, y))

# Muestra los números de forma aleatoria
sample = np.random.choice(X.shape[0], 10)
plt.imshow(X[sample, :].reshape(-1, 20).T)
plt.axis('off')
plt.show()