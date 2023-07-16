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
models_psi = define_model_psi_dict(ode_systems, target_folder,estimation=True)
print(models_psi)
models_func = {1: model1,
               2: model2}

# read data
data = pd.read_csv('./data/data.csv')
data = data.sort_values(by='T')
Y = data.values[:,1:]
T = data['T'].values

# estimate parameters
models_best_w = {}

for m in models_func.keys():
    print('MLE for Model {}'.format(m))
    models_best_w[m] = estimate_model_params(m, Y, T, models_func, models_psi)

print(models_best_w)
X_hat_A = None
X_hat_B = None
for m in models_best_w.keys():
    xi = models_best_w[m]['xi']
    psi = models_best_w[m]['psi']
    X_hat = odeint(models_func[m],
                   xi,
                   T,
                   args=tuple(list(psi)))
    if m == 1:
        X_hat_A = X_hat.copy()
    else:
        X_hat_B = X_hat.copy()

# plot models curves
for c in range(Y.shape[1]):
    plt.plot(T, Y[:,c], label=r'$Y_{}$'.format(c))
    plt.plot(T, X_hat_A[:,c], label=r'$\hat{x}_A$')
    plt.plot(T, X_hat_B[:,c], label=r'$\hat{x}_B$')
    plt.legend(fontsize=14)
    plt.xlabel('Olsen P', fontsize=14)
    plt.ylabel('Yield', fontsize=14)
    plt.show()

# create model comparison setup table
theta_setups = create_thetas_df(Y, T, models_best_w, models_psi, models_func)
theta_setups.to_csv('./data/theta_setups.csv', index=False)
theta_setups = pd.read_csv('./data/theta_setups.csv')

# run model selection
model_selection_table = selection_in_favor(theta_setups,
                                           ode_systems,
                                           target_folder,
                                           models_psi,
                                           Y,
                                           T,
                                           alpha,
                                           z_half_alpha)
print(model_selection_table)
# ### End ###