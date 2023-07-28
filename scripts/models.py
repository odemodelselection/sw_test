import numpy as np

# Define ODE models in scipy.integrate.odeint format where model1, model2,...,modelK
# corresponds to the model_1.txt,model_2.txt,...,model_K.txt files in models folder.
# Gompertz
def model1(x, t, psi1, psi2):
    x1 = x
    d_x1 = -psi1*x1*np.log(x1/psi2)
    return np.array(d_x1)

#ExpDecay + bias
def model2(x, t, psi1, psi2):
    x1 = x
    d_x1 = psi1*x1 + psi2
    return np.array(d_x1)

#Logistic Growth
def model3(x, t, psi1, psi2):
    x1 = x
    d_x1 = psi1*x1*(1 - x1/psi2)
    return np.array(d_x1)

# #Von Bertalanffy model - obtained MLE later produce errors in casadi when calculating derivatives
# def model4(x, t, psi1, psi2, psi3):
#     x1 = x
#     d_x1 = psi1*x1**(psi3) - psi2*x1
#     return np.array(d_x1)

#Richardson model
def model4(x, t, psi1, psi2, psi3):
    x1 = x
    d_x1 = psi1*(1-(x1/psi2)**psi3)*x1
    return np.array(d_x1)

