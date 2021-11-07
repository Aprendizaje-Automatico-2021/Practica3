import numpy as np
from numpy.lib import diff
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

def sigmoide_fun(Z):
    return 1 / (1 + (np.exp(-Z)))

def getEtiqueta(Y, etiqueta):
    """
    Devuelve el vector de booleanos para determinar
    que se trata de la etiqueta correcta
    """
    y_etiqueta = np.ravel(Y) == etiqueta # Vector de booleanos
    y_etiqueta = y_etiqueta * 1 # Conversión de bool a 0|1
    return y_etiqueta   # (5K,)

def coste(Theta, X, Y, lamb):
    """
        Calculates the J function of the cost
        of the Logistic Regresion    
    """
    m = X.shape[0]  # m = 5K
    S = sigmoide_fun(np.matmul(X, Theta)) # (5K,)
    Sum1 = np.dot(Y, np.log(S)) # 

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

def gradiente(theta, X, Y, lamb):
    """
        Calculate the new value of Theta with matrix
    """
    m = X.shape[0]  # m = 5K
    S = sigmoide_fun(np.matmul(X, theta))   # (5K,)
    diff = S - Y # Y.shape = (5K,)
    newTheta = (1 / m) * np.matmul(X.T, diff) + (lamb/m) * theta
    newTheta[0] -= (lamb/m) * theta[0]

    return newTheta

def evalua(i, theta, X, Y): 
    S = sigmoide_fun(np.matmul(X, theta))   # (5K,)
    pos = np.where(S >= 0.5)   #(5K,)
    neg = np.where(S < 0.5) #(5K,)
    posExample = np.where(Y == 1)
    negExample = np.where(Y == 0)

    # intersect1d: sirve para coger añadir elementos al vector
    # cuando éstos sean iguales
    totalPos = np.intersect1d(pos, posExample).shape[0] / S.shape[0]
    totalNeg = np.intersect1d(neg, negExample).shape[0] / S.shape[0]
    # El porcentaje total sale de la cantidad de ejemplos identificados
    # como la etiqueta y de la cantidad que ha identifiado que no son la etiqueta
    print("Total {}: {}%".format(i, (totalPos + totalNeg) * 100))
    return totalPos + totalNeg

def oneVsAll(X, y, num_etiquetas, reg):
    """
    Entrenamiento de varios clasificadores por regresión logística
    """
    m = X.shape[1]  # m = 400
    theta = np.zeros((num_etiquetas, m)) # (10, 400)
    y_etiquetas = np.zeros((y.shape[0], num_etiquetas)) # (5K, 10)

    for i in range(num_etiquetas):
        y_etiquetas[:, i] = getEtiqueta(y, i)
    # Las etiquetas de los 0's
    y_etiquetas[:, 0] = getEtiqueta(y, 10)

    for i in range(num_etiquetas):
        # Cálculo del theta óptimo para cada ejemplo de entrenamiento, para cada etiqueta
        # theta[i, :] -> (400,) para calcular el theta óptimo de cada px de cada img
        result = opt.fmin_tnc(func = coste, x0 = theta[i, :], fprime = gradiente,
                args=(X, y_etiquetas[:, i], reg))
        theta[i, :] = result[0]

    # Evaluación con el valor óptimo (theta[i, :])
    # de cada etiqueta y_etiquetas[:, i]. 
    evaluacion = np.zeros(num_etiquetas) # (10,)
    for i in range(num_etiquetas):
        evaluacion[i] = evalua(i, theta[i, :], X, y_etiquetas[:, i])
    print("Evaluación media: ", evaluacion.mean() * 100)
    return 0

#main
data = loadmat ('./src/ex3data1.mat')
y = data ['y']  # (5K, 1)
X = data ['X']  # (5K, 400)

all_thetas = oneVsAll(X, y, 10, 0.1)

# Muestra los números de forma aleatoria
sample = np.random.choice(X.shape[0], 10)
plt.imshow(X[sample, :].reshape(-1, 20).T)
plt.axis('off')
plt.show()
