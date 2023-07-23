import os
import sys
import inspect
import glob
import matplotlib.colors as mcolors
named_colors = list(mcolors.CSS4_COLORS.keys())

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + '/scripts/')

from sw_test import *
from models import *

files = glob.glob('./plots/*')
for f in files:
    os.remove(f)
### Code to change ###
# Desired alpha-level of controlling the size of the test
alpha = 0.05
log_transform = False
B = 1000
BB = 100
### Don't change this ###
z_half_alpha = stats.norm.ppf(alpha/2, 0, 1)
# get names of .txt files in "models" folder
target_folder = './models/'
ode_systems = os.listdir(target_folder)
models_func = {i: eval('model{}'.format(i)) for i in range(1, len(ode_systems) + 1)}
models_psi = define_model_psi_dict(ode_systems, target_folder,estimation=True)

# read data
data = pd.read_csv('./data/data.csv')
data.columns = ['T'] + ['Y{}'.format(i) for i in range(1, data.shape[1])]
data = data.sort_values(by='T')
Y = data.values[:,1:]
if log_transform:
    Y = np.log(Y + 1)
T = data['T'].values
plt.plot(T, Y)
plt.show()
# estimate parameters
models_best_w = {}

for m in models_func.keys():
    print('MLE for Model {}'.format(m))
    models_best_w[m] = estimate_model_params(m, Y, T, models_func,
                                             models_psi, B=B, BB=BB, save_plots=True)

print(models_best_w)

# create model comparison setup table
theta_setups = create_thetas_df(Y, T, models_best_w, models_psi, models_func)
theta_setups.to_csv('./data/theta_setups.csv', index=False)
