from scipy import stats
from casadi import *
from scipy.integrate import odeint
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
import matplotlib.colors as mcolors
named_colors = list(mcolors.CSS4_COLORS.keys())
named_colors0 = plt.cm.tab10.colors
from models import *

## SW Test
class SWtest:
    def __init__(self):
        self.target_folder = './models/'
        self.ode_systems = os.listdir(self.target_folder)
        self.models_func = {i: eval('model{}'.format(i)) for i in range(1, len(self.ode_systems) + 1)}
        self.Y = None
        self.T = None

    def preprocess_txt(self, x):
        return self.remove_punctuation(x.replace('\n', '').strip().lower()).split()

    def remove_punctuation(self, x):
        x = re.sub(r'[\W]', ' ', x)
        x = ' '.join(x.split())
        return x

    def x_replace(self, x):
        return x.replace(x.replace('x', ''), '[{}]'.format(int(x.replace('x', '')) - 1)).replace('x', 'x_symb')

    def theta_replace(self, theta):
        return theta.replace(theta.replace('theta', ''), '[{}]'.format(int(theta.replace('theta', '')) - 1)).replace(
            'theta', 'psi_symb')

    def casadi_replace(self, ode, x_vars_dict, theta_vars_dict):
        for x, new_x in x_vars_dict.items():
            ode = ode.replace(x, new_x)
        for theta, new_theta in theta_vars_dict.items():
            ode = ode.replace(theta, new_theta)

        return ode

    def get_sigma2(sefl, Y, Xhat):
        return np.power(np.std(Y - Xhat, axis=0), 2)

    def sw_test_vec(self, X_hat_A, X_hat_B, sigmas2_A, sigmas2_B,
                    Y_A, Y_B, V_A, H_A, V_B, H_B,
                    h=None, alpha=0.05, return_full=False):
        def sw_ll_k_i(Y, X_hat, sigmas2):
            log_koef = 0
            log_sum = 0
            for i in range(Y.shape[1]):
                if np.sum(np.power(Y[:, i] - X_hat[:, i], 2) / (2 * sigmas2[i])) != 0:
                    log_koef += np.log(1 / np.sqrt(2 * np.pi * sigmas2[i]))
                    log_sum += np.power(Y[:, i] - X_hat[:, i], 2) / (2 * sigmas2[i])

            return log_koef - log_sum

        def sw_d_tilta(ll_A_i, ll_B_i, h):
            di = []

            for i in range(1, len(ll_A_i) + 1):
                di1 = ((i - 1) % 2 * h + 1) * ll_A_i[i - 1]
                di2 = (i % 2 * h + 1) * ll_B_i[i - 1]
                di.append(di1 - di2)

            return np.mean(di)

        def sw_get_h(n, alpha, sigma2_hat, cov_M, V_A, H_A, V_B, H_B):
            sigma_hat = np.sqrt(sigma2_hat)
            z_alpha2 = stats.norm.ppf(alpha / 2, 0, 1)

            lmbda = (sigma_hat / 2) * (z_alpha2 - np.sqrt(4 + np.power(z_alpha2, 2)))

            # print("Discuss max for differently shaped H^-1@V matrices of A and B")
            Cpl = stats.norm.pdf(z_alpha2 - lmbda / sigma_hat, 0, 1) * (
                        lmbda * (sigma2_hat - 2 * (cov_M[0, 0] + cov_M[1, 1]))) / (4 * np.power(sigma_hat, 3))
            Csd = (2 * stats.norm.pdf(z_alpha2, 0, 1) * np.max([np.abs(np.trace(np.linalg.inv(H_A) @ V_A)),
                                                                np.abs(np.trace(np.linalg.inv(H_B) @ V_B))])) / np.sqrt(
                (cov_M[0, 0] + cov_M[1, 1]) / 2)
            h = np.power(Csd / Cpl, 1 / 3) * np.power(n, -1 / 6) * np.power(np.log(np.log(n)), 1 / 3)

            return h

        if len(Y_A) != len(Y_B):
            print('The length of Y_A is not equal to the shape of Y_B')
            sys.exit(1)

        n = len(Y_A)

        ll_A_i = sw_ll_k_i(Y_A, X_hat_A, sigmas2_A)
        ll_B_i = sw_ll_k_i(Y_B, X_hat_B, sigmas2_B)

        cov_M = np.cov(ll_A_i, ll_B_i, bias=True)
        sigma2_hat = cov_M[0, 0] + cov_M[1, 1] - 2 * cov_M[0, 1]

        if h is None:
            h = sw_get_h(n, alpha, sigma2_hat, cov_M, V_A, H_A, V_B, H_B)

        # print(h)
        sigma2_tilda = (1 + h) * sigma2_hat + (np.power(h, 2) / 2) * (cov_M[0, 0] + cov_M[1, 1])
        d_tilta = sw_d_tilta(ll_A_i, ll_B_i, h)

        if return_full:
            return (np.sqrt(n) * d_tilta) / np.sqrt(sigma2_tilda), np.sqrt(sigma2_tilda), d_tilta, np.sqrt(n)
        else:
            return (np.sqrt(n) * d_tilta) / np.sqrt(sigma2_tilda)

    def get_J_H(self, dae, cur_t, psi_hat, xi_hat, hat_sigma2, Y, Xhat, i):
        d = len(xi_hat)
        p = len(psi_hat)

        g_i = Y[i, :] - Xhat[i, :]
        hat_sigma2 = 1 / hat_sigma2

        options = {'reltol': 1e-10, 'abstol': 1e-12, 'tf': cur_t,
                   'output_t0': False, 'max_num_steps': 500000, 'verbose': False,
                   'dump_in': False, 'dump_out': False,
                   'dump_dir': 'C:/DrugsIntel/SummerPracticum/Thesis/NoteBooks/casadi/pipeline_folder/'}
        F = integrator('F', 'cvodes', dae, options)

        D_x = F.factory('D', ['x0', 'p'], ['jac:xf:x0'])
        D_p = F.factory('D', ['x0', 'p'], ['jac:xf:p'])

        # Solve the problem
        D_xi = D_x(x0=xi_hat, p=psi_hat)['jac_xf_x0'].full()
        D_psi = D_p(x0=xi_hat, p=psi_hat)['jac_xf_p'].full()

        # Jacobian
        nabla_ln_f_sigma = (np.power(g_i, 2) * np.power(hat_sigma2, 2) - hat_sigma2) / 2
        nabla_ln_f_xi = D_xi.T @ (g_i * hat_sigma2)
        nabla_ln_f_psi = D_psi.T @ (g_i * hat_sigma2)

        Jacobian = np.hstack([nabla_ln_f_sigma, nabla_ln_f_xi, nabla_ln_f_psi])

        # Hessian
        nabla2_ln_f_sigma_sigma = np.diag(np.power(hat_sigma2, 2) / 2 - np.power(g_i, 2) * np.power(hat_sigma2, 3))
        nabla2_ln_f_sigma_xi = -np.diag(g_i * np.power(hat_sigma2, 2)) @ D_xi
        nabla2_ln_f_sigma_psi = -np.diag(g_i * np.power(hat_sigma2, 2)) @ D_psi

        nabla2_ln_f_sigma = np.hstack([nabla2_ln_f_sigma_sigma,
                                       nabla2_ln_f_sigma_xi,
                                       nabla2_ln_f_sigma_psi])

        H_x_x = D_x.factory('H', ['x0', 'p'], ['jac:jac_xf_x0:x0'])
        H_x_p = D_x.factory('H', ['x0', 'p'], ['jac:jac_xf_x0:p'])
        H_p_x = D_p.factory('H', ['x0', 'p'], ['jac:jac_xf_p:x0'])
        H_p_p = D_p.factory('H', ['x0', 'p'], ['jac:jac_xf_p:p'])

        # Solve the problem
        D_xi_xi = H_x_x(x0=xi_hat, p=psi_hat)['jac_jac_xf_x0_x0'].full()
        D_xi_psi = H_x_p(x0=xi_hat, p=psi_hat)['jac_jac_xf_x0_p'].full()
        D_psi_xi = H_p_x(x0=xi_hat, p=psi_hat)['jac_jac_xf_p_x0'].full()
        D_psi_psi = H_p_p(x0=xi_hat, p=psi_hat)['jac_jac_xf_p_p'].full()

        # V2 matrices
        V_2_xi_xi = np.full([d, d], np.nan)
        steps = np.arange(d)
        for j in range(d):
            cur_rows = j * d + steps
            for jj in range(d):
                V_2_xi_xi[j, jj] = D_xi_xi[cur_rows, jj].T @ (g_i * hat_sigma2)

        V_2_xi_psi = np.full([d, p], np.nan)
        for j in range(d):
            cur_rows = j * d + steps
            for jj in range(p):
                V_2_xi_psi[j, jj] = D_xi_psi[cur_rows, jj].T @ (g_i * hat_sigma2)

        V_2_psi_xi = np.full([p, d], np.nan)
        for j in range(p):
            cur_rows = j * d + steps
            for jj in range(d):
                V_2_psi_xi[j, jj] = D_psi_xi[cur_rows, jj].T @ (g_i * hat_sigma2)

        V_2_psi_psi = np.full([p, p], np.nan)
        for j in range(p):
            cur_rows = j * d + steps
            for jj in range(p):
                V_2_psi_psi[j, jj] = D_psi_psi[cur_rows, jj].T @ (g_i * hat_sigma2)

        # V matrices
        V_xi_xi = np.full([d, d], np.nan)
        for j in range(d):
            for jj in range(d):
                V_xi_xi[j, jj] = (D_xi[j, :] * D_xi[jj, :]).T @ hat_sigma2

        V_xi_psi = np.full([d, p], np.nan)
        for j in range(d):
            for jj in range(p):
                V_xi_psi[j, jj] = (D_xi[:, j] * D_psi[:, jj]).T @ hat_sigma2

        V_psi_xi = np.full([p, d], np.nan)
        for j in range(p):
            for jj in range(d):
                V_psi_xi[j, jj] = (D_psi[:, j] * D_xi[:, jj]).T @ hat_sigma2

        V_psi_psi = np.full([p, p], np.nan)
        for j in range(p):
            for jj in range(p):
                V_psi_psi[j, jj] = (D_psi[:, j] * D_psi[:, jj]).T @ hat_sigma2

        nabla2_ln_f_xi_sigma = -D_xi @ np.diag(g_i * np.power(hat_sigma2, 2))
        nabla2_ln_f_xi_xi = V_2_xi_xi - V_xi_xi
        nabla2_ln_f_xi_psi = V_2_xi_psi - V_xi_psi

        nabla2_ln_f_xi = np.hstack([nabla2_ln_f_xi_sigma,
                                    nabla2_ln_f_xi_xi,
                                    nabla2_ln_f_xi_psi])

        nabla2_ln_f_psi_sigma = -D_psi.T @ np.diag(g_i * np.power(hat_sigma2, 2))
        nabla2_ln_f_psi_xi = V_2_psi_xi - V_psi_xi
        nabla2_ln_f_psi_psi = V_2_psi_psi - V_psi_psi

        nabla2_ln_f_psi = np.hstack([nabla2_ln_f_psi_sigma,
                                     nabla2_ln_f_psi_xi,
                                     nabla2_ln_f_psi_psi])

        Hessian = np.vstack([nabla2_ln_f_sigma,
                             nabla2_ln_f_xi,
                             nabla2_ln_f_psi])

        return Jacobian, Hessian

    def get_SW_HV(self, dae, T, xi_hat, psi_hat, sigma2_hat, Y, estimated, s):
        options = dict(grid=T)

        F = integrator('F', 'cvodes', dae, options)
        r = F(x0=xi_hat, p=psi_hat)

        Xhat = []
        for i in range(len(xi_hat)):
            Xhat.append([xi_hat[i]] + vertsplit(r['xf'], 1)[i].info()['data'])

        Xhat = np.vstack(Xhat).T

        if s != 0:
            Y = np.hstack([Y, Xhat[:, -s:]])

        J_dim = len(sigma2_hat) + len(xi_hat) + len(psi_hat)
        J = np.full([1, J_dim, len(T)], np.nan)
        H = np.full([J_dim, J_dim, len(T)], np.nan)
        # for i in range(len(T)):
        for i in tqdm(range(len(T))):
            cur_t = T[i]
            cur_J, cur_H = self.get_J_H(dae, cur_t, psi_hat, xi_hat, sigma2_hat, Y, Xhat, i)
            J[:, :, i] = cur_J
            H[:, :, i] = cur_H

        J = np.mean(J, axis=2)
        J = J[:, estimated]
        H = np.mean(H, axis=2)
        H = H[estimated, :][:, estimated]

        V = J.reshape(-1, 1) @ J.reshape(-1, 1).T

        return V, H, Xhat

    def selection_in_favor(self,
                           theta_setups,
                           ode_systems,
                           target_folder,
                           models_psi,
                           Y,
                           T,
                           alpha,
                           z_half_alpha):

        theta_setups_idx = theta_setups['model'].unique()

        if len(theta_setups_idx) != len(ode_systems):
            print('Exit on Error of theta_setups_idx != ODE_systems')
            sys.exit(1)

        models_est_params = dict()

        for cur_system in range(len(ode_systems)):
            models_est_params[cur_system] = dict()

            cur_ODE_system = ode_systems[cur_system]
            model = int(cur_ODE_system.split('_')[-1].replace('.txt', ''))

            with open(target_folder + '/' + cur_ODE_system) as f:
                lines = f.readlines()
                lines = [line.replace('psi', 'theta') for line in lines]


            cur_theta_setup = theta_setups[(theta_setups['model'] == model)]
            sigma_to_exclude = cur_theta_setup[(cur_theta_setup['to_include'] != 1) &
                                               (cur_theta_setup['theta'].str.contains('sigma'))].shape[0]
            xi_to_exclude = cur_theta_setup[(cur_theta_setup['to_include'] != 1) &
                                            (cur_theta_setup['theta'].str.contains('xi'))].shape[0]

            if sigma_to_exclude != xi_to_exclude:
                print('Shape of sigma to exclude is not equal ' /
                      'to the shape of xi to exclude')
                sys.exit(1)

            s = sigma_to_exclude

            x_vars = ['x{}'.format(i) for i in range(1, len(lines) + 1)]
            x_vars_in_equations = []
            theta_vars = []

            for line in lines:
                cur_vars = self.preprocess_txt(line)
                for var in cur_vars:
                    if 'theta' in var:
                        theta_vars.append(var)
                    elif 'x' in var:
                        x_vars_in_equations.append(var)
                    else:
                        try:
                            int(var)
                        except:
                            print('Process stop due to the unknown variable: {}'.format(var))
                            sys.exit(1)

            x_vars_in_equations = sorted(set(x_vars_in_equations))

            if len(x_vars) - models_psi[model]['inactive_states_in_ODE'] != len(x_vars_in_equations):
                print('Warning! Number of lines in model`s file does not equal the number of xi variables in equations')
                sys.exit(1)

            theta_vars = sorted(set(theta_vars))

            x_vars_dict = {x: self.x_replace(x) for x in x_vars}
            theta_vars_dict = {theta: self.theta_replace(theta) for theta in theta_vars}
            d = len(x_vars_dict.keys())
            p = len(theta_vars_dict.keys())

            if d - s != Y.shape[1]:
                print('Shape of Y is not equal to the shape of ODE system')
                sys.exit(1)

            sigmas_setup = cur_theta_setup[cur_theta_setup['theta'].str.contains('sigma')]
            sigmas_setup = sigmas_setup.sort_values(by='theta')
            xi_setup = cur_theta_setup[cur_theta_setup['theta'].str.contains('xi')]
            xi_setup = xi_setup.sort_values(by='theta')
            psi_setup = cur_theta_setup[cur_theta_setup['theta'].str.contains('psi')]
            psi_setup = psi_setup.sort_values(by='theta')

            if d != sigmas_setup.shape[0]:
                print('Shape of sigma2 is not equal to the shape of ODE system')
                sys.exit(1)

            if d != xi_setup.shape[0]:
                print('Shape of xi is not equal to the shape of ODE system')
                sys.exit(1)

            if p != psi_setup.shape[0]:
                print('Shape of psi is not equal to the shape of ODE system')
                sys.exit(1)

            if np.sum(sigmas_setup['value'].isnull()) != 0:
                print('Missed sigma value:')
                print(sigmas_setup[sigmas_setup['value'].isnull()])
                sys.exit(1)

            if np.sum(xi_setup['value'].isnull()) != 0:
                print('Missed xi value:')
                print(xi_setup[xi_setup['value'].isnull()])
                sys.exit(1)

            if np.sum(psi_setup['value'].isnull()) != 0:
                print('Missed psi value:')
                print(psi_setup[psi_setup['value'].isnull()])
                sys.exit(1)

            cur_theta_setup = pd.concat([sigmas_setup, xi_setup, psi_setup])
            estimated = cur_theta_setup['estimated'].values == 1
            xi_hat = xi_setup['value'].values
            psi_hat = psi_setup['value'].values
            sigma2_hat = sigmas_setup['value'].values
            # Formulate the ODE
            global x_symb, psi_symb
            x_symb = SX.sym('x', d)
            psi_symb = SX.sym('p', p)
            f = vertcat(*(eval(self.casadi_replace(ode, x_vars_dict, theta_vars_dict)) for ode in lines))
            dae = dict(x=x_symb, p=psi_symb, ode=f)

            V, H, Xhat = self.get_SW_HV(dae, T, xi_hat, psi_hat, sigma2_hat, Y, estimated, s)
            models_est_params[cur_system]['Xhat'] = Xhat
            models_est_params[cur_system]['Y'] = Y
            models_est_params[cur_system]['sigmas2_hat'] = sigma2_hat
            models_est_params[cur_system]['Vhat'] = V
            models_est_params[cur_system]['Hhat'] = H

        model_combs = list(product(list(models_est_params.keys()), list(models_est_params.keys())))
        model_combs = list(set([tuple(sorted(x)) for x in model_combs if x[0] != x[1]]))

        best_model = []
        sw_values = []
        modelA_sigma2_sums = []
        modelB_sigma2_sums = []
        for cur_models in model_combs:
            modelA = cur_models[0]
            modelB = cur_models[1]

            Y_A = models_est_params[modelA]['Y']
            X_hat_A = models_est_params[modelA]['Xhat']
            sigmas2_A = models_est_params[modelA]['sigmas2_hat']
            modelA_sigma2_sums.append(np.sum(sigmas2_A))
            V_A = models_est_params[modelA]['Vhat']
            H_A = models_est_params[modelA]['Hhat']

            Y_B = models_est_params[modelB]['Y']
            X_hat_B = models_est_params[modelB]['Xhat']

            sigmas2_B = models_est_params[modelB]['sigmas2_hat']
            modelB_sigma2_sums.append(np.sum(sigmas2_B))
            V_B = models_est_params[modelB]['Vhat']
            H_B = models_est_params[modelB]['Hhat']

            sw_value = self.sw_test_vec(X_hat_A, X_hat_B, sigmas2_A, sigmas2_B, Y_A, Y_B, V_A, H_A, V_B, H_B, h=None,
                                   alpha=alpha)
            sw_values.append(sw_value)
            if sw_value > np.abs(z_half_alpha):
                cur_best_model = ode_systems[modelA]
            elif sw_value < -np.abs(z_half_alpha):
                cur_best_model = ode_systems[modelB]
            else:
                cur_best_model = None
            best_model.append(cur_best_model)

        model_selection = pd.DataFrame(model_combs, columns=['model A', 'model B']) + 1
        model_selection['sw_value'] = sw_values
        model_selection['in favor'] = [int(x.split('_')[-1].replace('.txt', '')) if x is not None else '-' for x in
                                       best_model]

        return model_selection

    def define_model_psi_dict(self,
                              ode_systems,
                              target_folder):
        models_psi = {}

        for ode in ode_systems:
            model = int(ode.split('_')[-1].replace('.txt', ''))
            models_psi[model] = {}

            with open(target_folder + '/' + ode) as f:
                lines = f.readlines()
                lines = [line.replace('psi', 'theta') for line in lines]

            x_vars_in_equations = []
            theta_vars = []

            for line in lines:
                cur_vars = self.preprocess_txt(line)
                for var in cur_vars:
                    if 'theta' in var:
                        theta_vars.append(var)
                    elif 'x' in var:
                        x_vars_in_equations.append(var)
                    else:
                        try:
                            int(var)
                        except:
                            print('Process stop due to the unknown variable: {}'.format(var))
                            sys.exit(1)

            x_vars_in_equations = sorted(set(x_vars_in_equations))

            models_psi[model]['inactive_states_in_ODE'] = len(lines) - len(x_vars_in_equations)

        return models_psi

    def get_model_selection_table(self,
                                  alpha=0.05,
                                  log_transform=False,
                                  n_plot=None):

        z_half_alpha = stats.norm.ppf(alpha / 2, 0, 1)
        models_psi = self.define_model_psi_dict(self.ode_systems, self.target_folder)

        data = pd.read_csv('./data/data.csv')
        cols = data.columns.tolist()
        state_names = cols[1:]
        time_name = cols[0]
        data.columns = ['T'] + ['Y{}'.format(i) for i in range(1, data.shape[1])]
        data = data.sort_values(by='T')
        Y = data.values[:, 1:]
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
            step = (T[-1] - T[0]) / n_plot
            ode_T = np.arange(T[0], T[-1] + step, step=step)
        else:
            ode_T = T.copy()

        for m in models_best_w.keys():
            xi = models_best_w[m]['xi']
            psi = models_best_w[m]['psi']
            X_hats[m] = odeint(self.models_func[m],
                               xi,
                               ode_T,
                               args=tuple(list(psi)))

        # plot models curves
        for c in range(Y.shape[1]):
            plt.scatter(T, Y[:, c], label=r'$Y_{}$'.format(c + 1))
            for m in X_hats.keys():
                plt.plot(ode_T, X_hats[m][:, c], label=r'$\hat{x}_{' + '{}'.format(str(m) + str(c + 1)) + '}$')
            plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel(time_name, fontsize=14)
            plt.ylabel(state_names[c], fontsize=14)
            try:
                os.remove('./plots/state_{}.png'.format(state_names[c]))
            except:
                pass
            plt.savefig('./plots/state_{}.png'.format(state_names[c]),
                        bbox_inches='tight')
            plt.show()

        for m in X_hats.keys():
            for c in range(Y.shape[1]):
                plt.plot(ode_T, X_hats[m][:, c], label=r'$\hat{x}_{' + '{}'.format(str(m) + str(c + 1)) + '}$',
                         c=named_colors[c])
                plt.scatter(T, Y[:, c], label=r'$Y_{}$'.format(c + 1), c=named_colors[c])
                plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlabel(time_name, fontsize=14)
                # plt.ylabel('State values', fontsize=14)
            try:
                os.remove('./plots/model_{}.png'.format(m))
            except:
                pass
            plt.savefig('./plots/model_{}.png'.format(m),
                        bbox_inches='tight')
            plt.show()

        model_selection_table = self.selection_in_favor(theta_setups,
                                                        self.ode_systems,
                                                        self.target_folder,
                                                        models_psi,
                                                        Y,
                                                        T,
                                                        alpha,
                                                        z_half_alpha)
        return model_selection_table


class Estimate:
    def __init__(self):
        self.B = None
        self.BB = None
        self.save_plots = None
        self.target_folder = './models/'
        self.ode_systems = os.listdir(self.target_folder)
        self.models_func = {i: eval('model{}'.format(i)) for i in range(1, len(self.ode_systems) + 1)}
        self.Y = None
        self.T = None
        self.models_psi = None
        self.models_best_w = None

    def preprocess_txt(self, x):
        return self.remove_punctuation(x.replace('\n', '').strip().lower()).split()

    def remove_punctuation(self, x):
        x = re.sub(r'[\W]', ' ', x)
        x = ' '.join(x.split())
        return x

    def create_thetas_df(self):
        all_thetas_dfs = []
        for cur_m in self.models_func.keys():
            print('Create theta_setup for model# {}'.format(cur_m))
            sigma_given = self.models_psi[cur_m]['sigma_given']
            xi_given = self.models_psi[cur_m]['xi_given']
            psi_given = self.models_psi[cur_m]['psi_given']
            best_w = self.models_best_w[cur_m]
            xi = best_w['xi']
            psi = best_w['psi']

            res = odeint(self.models_func[cur_m], xi, self.T, args=tuple(psi))

            sigmas2_hat = []

            for c in range(self.Y.shape[1]):
                cur_X = res[:, c]
                if sigma_given[c] != sigma_given[c]:
                    sigmas2_hat.append(np.power(np.std(self.Y[:, c] - cur_X), 2))
                else:
                    sigmas2_hat.append(sigma_given[c])

            sigmas2_hat = sigmas2_hat + [1] * (len(sigma_given) - len(sigmas2_hat))
            xi = xi + [0] * (len(xi_given) - len(xi))
            thetas = [list(sigmas2_hat) + list(xi) + list(psi)]

            thetas_df = pd.DataFrame(thetas, columns=
            ['sigma{}'.format(i) for i in range(1, len(sigmas2_hat) + 1)] +
            ['xi{}'.format(i) for i in range(1, len(xi) + 1)] +
            ['psi{}'.format(i) for i in range(1, len(psi) + 1)])
            thetas_df = pd.DataFrame(thetas_df.T.unstack()).reset_index().drop(columns=['level_0'])
            thetas_df.columns = ['theta', 'value']
            thetas_df['model'] = cur_m
            sigma_est = [1 if sigma_given[c] != sigma_given[c] else 0 for c in range(self.Y.shape[1])]
            sigma_est = sigma_est + [0] * (len(sigma_given) - len(sigma_est))
            xi_est = [1 if xi_given[c] != xi_given[c] else 0 for c in range(self.Y.shape[1])]
            xi_est = xi_est + [0] *  (len(xi_given) - len(xi_est))
            psi_est = [1 if psi_given[c] != psi_given[c] else 0 for c in range(len(psi_given))]
            thetas_df['estimated'] = sigma_est + xi_est + psi_est
            thetas_df['to_include'] = [1] * self.Y.shape[1] + [0] * (len(sigma_given) - self.Y.shape[1]) + \
                                      [1] * self.Y.shape[1] + [0] * (len(xi_given) - self.Y.shape[1]) + \
                                      [1] * len(psi_given)
            all_thetas_dfs.append(thetas_df)

        return pd.concat(all_thetas_dfs)

    def define_model_psi_dict(self):
        models_psi = {}

        for ode in self.ode_systems:
            model = int(ode.split('_')[-1].replace('.txt', ''))
            models_psi[model] = {}

            with open(self.target_folder + '/' + ode) as f:
                lines = f.readlines()
                lines = [line.replace('psi', 'theta') for line in lines]

            x_vars_in_equations = []
            theta_vars = []

            for line in lines:
                cur_vars = self.preprocess_txt(line)
                for var in cur_vars:
                    if 'theta' in var:
                        theta_vars.append(var)
                    elif 'x' in var:
                        x_vars_in_equations.append(var)
                    else:
                        try:
                            int(var)
                        except:
                            print('Process stop due to the unknown variable: {}'.format(var))
                            sys.exit(1)

            x_vars_in_equations = sorted(set(x_vars_in_equations))

            models_psi[model]['inactive_states_in_ODE'] = len(lines) - len(x_vars_in_equations)

            est_setups = pd.read_csv('./data/estimation_setups.csv')
            model_est_setups = est_setups[est_setups['model'] == model]
            x_vars_in_equations = sorted(set(x_vars_in_equations))
            theta_vars = sorted(set(theta_vars))

            models_psi[model]['p'] = len(x_vars_in_equations)
            models_psi[model]['d'] = len(theta_vars)

            sigma_est_setups = model_est_setups[model_est_setups['parameter'].str.contains('sigma')]
            xi_est_setups = model_est_setups[model_est_setups['parameter'].str.contains('xi')]
            psi_est_setups = model_est_setups[model_est_setups['parameter'].str.contains('psi')]

            if len(sigma_est_setups) != len(xi_est_setups):
                print(
                    'Exit on Error of the number of sigma != the number of xi for model {} in theta_setups.csv'.format(
                        model))
                sys.exit(1)

            if len(x_vars_in_equations) != len(xi_est_setups):
                print('Exit on Error of x_vars_in_equations != xi_est_setups')
                sys.exit(1)

            sigma_given = []
            for i in range(len(x_vars_in_equations)):
                cur_sigma = sigma_est_setups[sigma_est_setups['parameter'] == 'sigma{}'.format(i + 1)]
                sigma_given.append(cur_sigma['given'].iloc[0])

            xi_given = []
            xi_bounds = []
            xi_inits = []
            for i in range(len(x_vars_in_equations)):
                cur_xi = xi_est_setups[xi_est_setups['parameter'] == 'xi{}'.format(i + 1)]
                if cur_xi['given'].iloc[0] != cur_xi['given'].iloc[0]:
                    lb = cur_xi['lower_bound'].iloc[0]
                    ub = cur_xi['upper_bound'].iloc[0]
                    if lb == 'infty' or lb == '-infty':
                        lb = -np.inf
                    if ub == 'infty':
                        ub = np.inf
                    li = cur_xi['lower_initial_value'].iloc[0]
                    ui = cur_xi['upper_initial_value'].iloc[0]
                    if li == li and ui == ui:
                        if li == 'infty' or li == '-infty':
                            li = -np.inf
                        if ui == 'infty':
                            ui = np.inf
                    else:
                        if float(lb) < 0 < float(ub):
                            if float(lb) > -1:
                                li = lb
                            else:
                                li = -1
                            if float(ub) < 1:
                                ui = ub
                            else:
                                ui = 1
                        else:
                            if 0 > float(lb):
                                if float(lb) > -1:
                                    li = lb
                                else:
                                    li = -1
                                ui = 0
                            else:
                                if float(ub) < 1:
                                    ui = ub
                                else:
                                    ui = 1
                                li = 0

                    xi_bounds.append((float(lb), float(ub)))
                    xi_given.append(np.NaN)
                    xi_inits.append((float(li), float(ui)))
                else:
                    xi_given.append(cur_xi['given'].iloc[0])
                    xi_bounds.append((np.NaN, np.NaN))
                    xi_inits.append((np.NaN, np.NaN))

            models_psi[model]['xi_est'] = np.sum([True if x != x else False for x in xi_given])
            models_psi[model]['xi_given'] = xi_given
            models_psi[model]['xi_bounds'] = xi_bounds
            models_psi[model]['xi_inits'] = xi_inits

            if len(theta_vars) != len(psi_est_setups):
                print('Exit on Error of theta_vars != psi_est_setups')
                sys.exit(1)

            psi_given = []
            psi_bounds = []
            psi_inits = []
            for i in range(len(psi_est_setups)):
                cur_psi = psi_est_setups[psi_est_setups['parameter'] == 'psi{}'.format(i + 1)]
                if cur_psi['given'].iloc[0] != cur_psi['given'].iloc[0]:
                    lb = cur_psi['lower_bound'].iloc[0]
                    ub = cur_psi['upper_bound'].iloc[0]
                    if lb == 'infty' or lb == '-infty':
                        lb = -np.inf
                    if ub == 'infty':
                        ub = np.inf
                    li = cur_psi['lower_initial_value'].iloc[0]
                    ui = cur_psi['upper_initial_value'].iloc[0]
                    if li == li and ui == ui:
                        if li == 'infty' or li == '-infty':
                            li = -np.inf
                        if ui == 'infty':
                            ui = np.inf
                    else:
                        if float(lb) < 0 < float(ub):
                            if float(lb) > -1:
                                li = lb
                            else:
                                li = -1
                            if float(ub) < 1:
                                ui = ub
                            else:
                                ui = 1
                        else:
                            if 0 > float(lb):
                                if float(lb) > -1:
                                    li = lb
                                else:
                                    li = -1
                                ui = 0
                            else:
                                if float(ub) < 1:
                                    ui = ub
                                else:
                                    ui = 1
                                li = 0
                    psi_bounds.append((float(lb), float(ub)))
                    psi_given.append(np.NaN)
                    psi_inits.append((float(li), float(ui)))
                else:
                    psi_given.append(cur_psi['given'].iloc[0])
                    psi_bounds.append((np.NaN, np.NaN))
                    psi_inits.append((np.NaN, np.NaN))

            models_psi[model]['psi_est'] = np.sum([True if x != x else False for x in psi_given])
            models_psi[model]['psi_given'] = psi_given
            models_psi[model]['psi_bounds'] = psi_bounds
            models_psi[model]['psi_inits'] = psi_inits
            models_psi[model]['sigma_given'] = sigma_given

        return models_psi

    def estimate_model_params(self,m):
        models_psi = self.define_model_psi_dict()

        p = models_psi[m]['psi_est']
        d = models_psi[m]['xi_est']

        def f(opt_w):
            xi = models_psi[m]['xi_given'].copy()
            ii = 0
            for i in range(len(xi)):
                if xi[i] != xi[i]:
                    xi[i] = opt_w[ii]
                    ii += 1
            psi = models_psi[m]['psi_given'].copy()
            for i in range(len(psi)):
                if psi[i] != psi[i]:
                    psi[i] = opt_w[ii]
                    ii += 1

            res = odeint(self.models_func[m], xi, self.T, args=tuple(psi))

            error = 0
            for c in range(self.Y.shape[1]):
                cur_X = res.T[c]
                error += np.mean(np.power(self.Y[:, c] - cur_X, 2))
            return error

        all_res = {}
        all_w = {}
        best_res = np.inf
        best_res_diff = np.inf
        best_res_s = []
        best_res_n = []

        for l in tqdm(range(self.B)):
            opt_w = []
            for i in range(len(models_psi[m]['xi_inits'])):
                if models_psi[m]['xi_given'][i] != models_psi[m]['xi_given'][i]:
                    opt_w.append(np.random.uniform(models_psi[m]['xi_inits'][i][0],
                                                   models_psi[m]['xi_inits'][i][1]))
            for i in range(len(models_psi[m]['psi_inits'])):
                if models_psi[m]['psi_given'][i] != models_psi[m]['psi_given'][i]:
                    opt_w.append(np.random.uniform(models_psi[m]['psi_inits'][i][0],
                                                   models_psi[m]['psi_inits'][i][1]))
            opt_w = np.asarray(opt_w)

            xi_bounds = [x for x, y in zip(models_psi[m]['xi_bounds'], models_psi[m]['xi_given']) if y != y]
            psi_bounds = [x for x, y in zip(models_psi[m]['psi_bounds'], models_psi[m]['psi_given']) if y != y]
            bounds = xi_bounds + psi_bounds
            f(opt_w)
            # run SLPSQ procedure
            res = minimize(f, opt_w,
                           bounds=bounds,
                           method='SLSQP',
                           options={'maxiter': 1000, 'ftol': 1e-6})

            if res['fun'] < best_res:
                if best_res_diff == np.inf:
                    best_res_diff = res['fun']
                else:
                    best_res_diff = best_res - res['fun']
                best_res = res['fun']

            best_res_s.append(best_res)
            best_res_n.append(l)

            all_res[l] = res['fun']
            all_w[l] = res['x']
            all_w[l][all_w[l] == 0] = 0.000001

        all_res_df = pd.DataFrame.from_dict(all_res, orient='index', columns=['MSE'])
        all_w_df = pd.DataFrame.from_dict(all_w, orient='index',
                                          columns=['coef_{}'.format(i) for i in range(p + d)])

        all_res_df = all_w_df.join(all_res_df)

        best_w = all_res_df[all_res_df['MSE'] == all_res_df['MSE'].min()].values[0, :-1]
        best_res = all_res_df[all_res_df['MSE'] == all_res_df['MSE'].min()].values[0, -1]

        for l in tqdm(range(self.BB)):
            opt_w = np.asarray(
                [stats.truncnorm.rvs(loc=best_w[i], scale=np.sqrt(np.abs(best_w[i])) / 3, a=0, b=np.inf)
                 for i in range(p + d)])
            f(opt_w)
            # run SLPSQ procedure
            res = minimize(f, opt_w,
                           bounds=bounds,
                           method='SLSQP',
                           options={'maxiter': 1000, 'ftol': 1e-6})

            if res['fun'] < best_res:
                if best_res_diff == np.inf:
                    best_res_diff = res['fun']
                else:
                    best_res_diff = best_res - res['fun']
                best_res = res['fun']
                best_w = res['x']

            best_res_s.append(best_res)
            best_res_n.append(l + self.B)

        xi = models_psi[m]['xi_given'].copy()
        ii = 0
        for i in range(len(xi)):
            if xi[i] != xi[i]:
                xi[i] = best_w[ii]
                ii += 1

        psi = models_psi[m]['psi_given'].copy()

        for i in range(len(psi)):
            if psi[i] != psi[i]:
                psi[i] = best_w[ii]
                ii += 1

        self.models_psi = models_psi
        plt.plot(best_res_n, np.log(best_res_s))
        plt.xlabel('iteration')
        plt.ylabel('min of loss function (log)')
        plt.title('Model: {}'.format(m))
        try:
            os.remove('./plots/loss_functionstate_{}.png'.format(m))
        except:
            pass
        plt.savefig('./plots/loss_functionstate_{}.png'.format(m), bbox_inches='tight')
        plt.clf()

        return {'xi': xi, 'psi': psi}

    def estimate_ODE_parameters(self,
                                log_transform=False,
                                B=1000,
                                BB=100):
        self.B = B
        self.BB = BB
        data = pd.read_csv('./data/data.csv')
        data.columns = ['T'] + ['Y{}'.format(i) for i in range(1, data.shape[1])]
        data = data.sort_values(by='T')
        self.Y = data.values[:, 1:]
        if log_transform:
            self.Y = np.log(self.Y + 1)
        self.T = data['T'].values

        self.models_best_w = {}

        for m in self.models_func.keys():
            print('MLE for Model {}'.format(m))
            self.models_best_w[m] = self.estimate_model_params(m)

        # create model comparison setup table
        theta_setups = self.create_thetas_df()
        theta_setups.to_csv('./data/theta_setups.csv', index=False)

class SWtestModelSelection:
    def __init__(self,
                 with_estimation=True,
                 alpha=0.05,
                 log_transform=False,
                 B=1000,
                 BB=100,
                 n_plot=None):
        self.with_estimation = with_estimation
        self.log_transform = log_transform
        self.B = B
        self.BB = BB
        self.alpha = alpha
        self.n_plot = n_plot

        self.Estimations = Estimate()
        self.RunSWtest = SWtest()

    def run(self):
        if self.with_estimation:
            self.Estimations.estimate_ODE_parameters(log_transform=self.log_transform,
                                                     B=self.B,
                                                     BB=self.BB)

        ms_table = self.RunSWtest.get_model_selection_table(alpha=self.alpha,
                                                            log_transform=self.log_transform,
                                                            n_plot=self.n_plot)

        ms_table.to_csv('./data/model_selection_results.csv', index=False)
        return ms_table
