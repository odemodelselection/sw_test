import os
import sys
import inspect
import warnings
warnings.filterwarnings("ignore")

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + '/scripts/')

from sw_test import SWtestModelSelection

sw_ms = SWtestModelSelection(with_estimation=False,
                             alpha=0.05,
                             log_transform=False,
                             B=500,
                             BB=100,
                             n_plot=200,
                             save_eps=True)

sw_ms_table = sw_ms.run()
print(sw_ms_table)
