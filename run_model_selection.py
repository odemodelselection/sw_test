import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + '/scripts/')

from sw_test import *

### Code to change ###
# Desired alpha-level of controlling the size of the test
alpha = 0.05
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
### End ###

### Don't change this ###
z_half_alpha = stats.norm.ppf(alpha/2, 0, 1)
# get names of .txt files in "models" folder
target_folder = './models/'
ode_systems = os.listdir(target_folder)
models_psi = define_model_psi_dict(ode_systems, target_folder)
models_func = {1: model1,
               2: model2}
# read data
data = pd.read_csv('./data/data.csv')
data = data.sort_values(by='T')
Y = data.values[:,1:]
T = data['T'].values
theta_setups = pd.read_csv('./data/theta_setups.csv')
model_selection_table = selection_in_favor(theta_setups,
                                           ode_systems,
                                           target_folder,
                                           models_psi,
                                           Y,
                                           T,
                                           alpha,
                                           z_half_alpha)
print(model_selection_table)
### End ###