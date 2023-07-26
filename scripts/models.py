import numpy as np

# Define ODE models in scipy.integrate.odeint format where model1, model2,...,modelK
# corresponds to the model_1.txt,model_2.txt,...,model_K.txt files in models folder.
def model1(x, t, psi1, psi2, psi3, psi4):
    x1, x2, x3 = x
    d_x1 = psi2*x3*x1 - psi4*x1 - psi3*x1
    d_x2 = psi4*x1 - psi3*x2
    d_x3 = psi1 - psi2 * x3 * x1 - psi3 * x3
    return np.array([d_x1, d_x2, d_x3])


def model2(x, t, psi1, psi2, psi3, psi4, psi5):
    x1, x2, x3, x4 = x
    d_x1 = psi5*x4 - psi4*x1 - psi3*x1
    d_x2 = psi4*x1 - psi3*x2
    d_x3 = psi1 - psi2*x3*x1 - psi3*x3
    d_x4 = psi2*x3*x1 - psi5*x4 - psi3*x4
    return np.array([d_x1, d_x2, d_x3, d_x4])


def model3(x, t, psi1, psi2, psi3, psi4, psi5):
    x1, x2, x3 = x
    d_x1 = psi2*x3*x1 - psi4*x1 - psi3*x1
    d_x2 = psi4*x1 - (psi3 + psi5)*x2
    d_x3 = psi1 - psi2*x3*x1 - psi3*x3 + psi5*x2
    return np.array([d_x1, d_x2, d_x3])
