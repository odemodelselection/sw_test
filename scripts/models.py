import numpy as np

# Define ODE models in scipy.integrate.odeint format where model1, model2,...,modelK
# corresponds to the model_1.txt,model_2.txt,...,model_K.txt files in models folder.
def model1(X, t, psi1, psi2):
    X1 = X
    dotX1 = psi1*(psi2 - X1)
    return np.array(dotX1)

def model2(X, t, psi1, psi2):
    X1 = X
    dotX1 = -psi1*np.power(-psi2 + X1, 2)
    return np.array(dotX1)