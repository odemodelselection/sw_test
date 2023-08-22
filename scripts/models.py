import numpy as np

# Define ODE models in scipy.integrate.odeint format where model1, model2,...,modelK
# corresponds to the model_1.txt,model_2.txt,...,model_K.txt files in models folder.
# eq(4)
def model4(x, t, psi1, psi2, psi3, psi4, psi5):
    x1, x2 = x
    d_x1 = psi2 * psi3 * x2 * x1 - psi4 * x1 - psi5 * np.power(x1, 2)
    d_x2 = psi1 * x2 - psi2 * x2 * x1
    return np.array([d_x1, d_x2])

# eq(3)
def model3(x, t, psi1, psi2, psi3, psi4, psi5):
    x1, x2 = x
    d_x1 = psi2 * psi3 * x2 * x1 / (1 + psi2 * psi5 * x1) - psi4 * x1
    d_x2 = psi1 * x2 - psi2 * x2 * x1 / (1 + psi2 * psi5 * x1)
    return np.array([d_x1, d_x2])

# eq(2)
def model2(x, t, psi1, psi2, psi3, psi4, psi5):
    x1, x2 = x
    d_x1 = psi2 * psi3 * x2 * x1 - psi4 * x1
    d_x2 = psi1 * x2 *(1 - x2/psi5) - psi2 * x2 * x1
    return np.array([d_x1, d_x2])

# eq(1)
def model1(x, t, psi1, psi2, psi3, psi4):
    x1, x2 = x
    d_x1 = psi2 * psi3 * x2 * x1 - psi4 * x1
    d_x2 = psi1 * x2 - psi2 * x2 * x1
    return np.array([d_x1, d_x2])
