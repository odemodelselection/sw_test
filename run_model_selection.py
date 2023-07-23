import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + '/scripts/')

from sw_test import *
from models import *

named_colors = plt.cm.tab10.colors
### Code to change ###
# Desired alpha-level of controlling the size of the test
alpha = 0.05
save_plots = True
log_transform = False
# define the number of time points to solve ODE for plots: keep None to use original T vector
n_plot = 100
### Don't change this ###
z_half_alpha = stats.norm.ppf(alpha/2, 0, 1)
# get names of .txt files in "models" folder
target_folder = './models/'
ode_systems = os.listdir(target_folder)
models_func = {i: eval('model{}'.format(i)) for i in range(1, len(ode_systems) + 1)}
models_psi = define_model_psi_dict(ode_systems, target_folder)

# read data
data = pd.read_csv('./data/data.csv')
cols = data.columns.tolist()
state_names = cols[1:]
time_name = cols[0]
data.columns = ['T'] + ['Y{}'.format(i) for i in range(1, data.shape[1])]
data = data.sort_values(by='T')
Y = data.values[:,1:]
if log_transform:
    Y = np.log(Y + 1)
T = data['T'].values
theta_setups = pd.read_csv('./data/theta_setups.csv')
models_best_w = {}
for m in theta_setups['model'].unique():
    theta_m = theta_setups[theta_setups['model'] == m]
    xi = theta_m[theta_m['theta'].str.contains('xi')]['value'].tolist()
    psi = theta_m[theta_m['theta'].str.contains('psi')]['value'].tolist()
    models_best_w[m] = {'xi': xi,
                        'psi': psi}

X_hats = {}
if n_plot is not None:
    step=(T[-1]-T[0])/n_plot
    ode_T = np.arange(T[0],T[-1]+step, step=step)
else:
    ode_T = T.copy()

for m in models_best_w.keys():
    xi = models_best_w[m]['xi']
    psi = models_best_w[m]['psi']
    X_hats[m] = odeint(models_func[m],
                       xi,
                       ode_T,
                       args=tuple(list(psi)))

# plot models curves
for c in range(Y.shape[1]):
    plt.scatter(T, Y[:, c], label=r'$Y_{}$'.format(c+1))
    for m in X_hats.keys():
        plt.plot(ode_T, X_hats[m][:,c], label=r'$\hat{x}_{' + '{}'.format(str(m) + str(c+1)) + '}$')
    plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(time_name, fontsize=14)
    plt.ylabel(state_names[c], fontsize=14)
    if save_plots:
        plt.savefig('./plots/state_{}.png'.format(state_names[c]),
                    bbox_inches='tight')
    plt.show()


for m in X_hats.keys():
    for c in range(Y.shape[1]):
        plt.plot(ode_T, X_hats[m][:,c], label=r'$\hat{x}_{' + '{}'.format(str(m) + str(c+1)) + '}$',
                 c=named_colors[c])
        plt.scatter(T, Y[:, c], label=r'$Y_{}$'.format(c+1), c=named_colors[c])
        plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(time_name, fontsize=14)
        # plt.ylabel('State values', fontsize=14)
    if save_plots:
        plt.savefig('./plots/model_{}.png'.format(m),
                    bbox_inches='tight')
    plt.show()

model_selection_table = selection_in_favor(theta_setups,
                                           ode_systems,
                                           target_folder,
                                           models_psi,
                                           Y,
                                           T,
                                           alpha,
                                           z_half_alpha)
print(model_selection_table)
## End ###