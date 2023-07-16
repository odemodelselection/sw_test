from scipy import stats
from casadi import *
from scipy.integrate import odeint
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import string
import os
from itertools import product

## SW Test
def preprocess_txt(x):
    return remove_punctuation(x.replace('\n', '').strip().lower()).split()

def remove_punctuation(x, remove_punc=string.punctuation):
    x = re.sub(r'[\W]', ' ', x)
    x = ' '.join(x.split())

    return x

def x_replace(x):
    return x.replace(x.replace('x', ''), '[{}]'.format(int(x.replace('x', '')) - 1)).replace('x', 'x_symb')

def theta_replace(theta):
    return theta.replace(theta.replace('theta', ''), '[{}]'.format(int(theta.replace('theta', '')) - 1)).replace(
        'theta', 'psi_symb')

def casadi_replace(ode, x_vars_dict, theta_vars_dict):
    for x, new_x in x_vars_dict.items():
        ode = ode.replace(x, new_x)
    for theta, new_theta in theta_vars_dict.items():
        ode = ode.replace(theta, new_theta)

    return ode

def scipy_replace(ode, x_vars_dict, theta_vars_dict):
    for x, new_x in x_vars_dict.items():
        ode = ode.replace(x, new_x).replace('x_symb', 'xi')
    for theta, new_theta in theta_vars_dict.items():
        ode = ode.replace(theta, new_theta).replace('_symb', '')

    return ode

def get_sigma2(Y, Xhat):
    return np.power(np.std(Y - Xhat, axis=0), 2)

# function that returns dy/dt
def model(xi, t, *psi):
    psi = list(psi)
    return tuple(eval(scipy_replace(ode, x_vars_dict, theta_vars_dict), {'psi': psi, 'xi': xi}) for ode in lines)

def xi_init(Y):
    xi_init = []
    for i in range(len(xi_bounds)):
        xi_init.append(np.random.normal(loc=Y[0, i], scale=np.std(Y[:, i])))

    return xi_init

def psi_init(psi_bounds):
    psi_init = []
    for i in range(len(psi_bounds)):
        lower_bound = psi_bounds[i][0]
        upper_bound = psi_bounds[i][1]
        if lower_bound > -np.inf and upper_bound < np.inf:
            psi_init.append(np.random.uniform(lower_bound, upper_bound))
        elif lower_bound == -np.inf and upper_bound < np.inf:
            psi_init.append(np.random.uniform(np.power(upper_bound, 1 / 3), upper_bound))
        elif lower_bound > -np.inf and upper_bound == np.inf:
            psi_init.append(np.random.uniform(lower_bound, np.power(lower_bound, 2)))
        else:
            psi_init.append(np.random.normal())

    return psi_init

def sw_test_vec(X_hat_A, X_hat_B, sigmas2_A, sigmas2_B,
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

def get_J_H(dae, cur_t, psi_hat, xi_hat, hat_sigma2, Y, Xhat, i):
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

def get_SW_HV(dae, T, xi_hat, psi_hat, sigma2_hat, Y, estimated, s, trapezoid_exp=False):
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
        cur_J, cur_H = get_J_H(dae, cur_t, psi_hat, xi_hat, sigma2_hat, Y, Xhat, i)
        J[:, :, i] = cur_J
        H[:, :, i] = cur_H

    J = np.mean(J, axis=2)
    J = J[:, estimated]
    H = np.mean(H, axis=2)
    H = H[estimated, :][:, estimated]

    V = J.reshape(-1, 1) @ J.reshape(-1, 1).T

    return V, H, Xhat


### Estimation
def estimate_model_params(m,
                          Y,
                          cur_time,
                          models_func,
                          models_psi,
                          B=1000,
                          BB=1000):
    p = models_psi[m]['psi_est']
    d = models_psi[m]['xi_est']

    def f(opt_w):
        # initial condition
        opt_w[opt_w == 0] = 0.000001
        xi = list(opt_w[:models_psi[m]['xi_est']]) + models_psi[m]['xi_given']
        psi = list(opt_w[models_psi[m]['xi_est']:]) + models_psi[m]['psi_given'] + models_psi[m]['add_psi']
        # print(xi, psi)
        res = odeint(models_func[m], xi, cur_time, args=tuple(psi))

        error = 0
        for c in range(Y.shape[1]):
            cur_X = res.T[c]
            error += np.mean(np.power(Y[:, c] - cur_X, 2))
        return error

    all_res = {}
    all_w = {}
    best_res = np.inf
    best_res_diff = np.inf
    diff_tol = 1e-6
    l = 0
    while best_res_diff > diff_tol and l < B:
        # initial uniform params
        opt_w = np.asarray([np.random.uniform(0, 1) for i in range(p + d)])
        bounds = models_psi[m]['xi_bounds'] + models_psi[m]['psi_bounds']
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
            # print(best_res_diff, best_res, l)

        all_res[l] = res['fun']
        all_w[l] = res['x']
        all_w[l][all_w[l] == 0] = 0.000001

        l += 1

    all_res_df = pd.DataFrame.from_dict(all_res, orient='index', columns=['MSE'])
    all_w_df = pd.DataFrame.from_dict(all_w, orient='index', columns=['coef_{}'.format(i) for i in range(p + d)])

    all_res_df = all_w_df.join(all_res_df)

    best_w = all_res_df[all_res_df['MSE'] == all_res_df['MSE'].min()].values[0, :-1]
    best_res = all_res_df[all_res_df['MSE'] == all_res_df['MSE'].min()].values[0, -1]

    l = 0

    while best_res_diff > diff_tol and l < BB:
        opt_w = np.asarray([stats.truncnorm.rvs(loc=best_w[i], scale=np.sqrt(best_w[i]) / 3, a=0, b=np.inf)
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
            best_w[best_w == 0] = 0.000001
            # print(best_res_diff, best_res, l)

        l += 1

    return best_w

def define_model_psi_dict(ode_systems,
                          target_folder):
    models_psi = {}

    for ode in ode_systems:
        model = int(ode.split('_')[-1].replace('.txt', ''))
        models_psi[model] = {}

        with open(target_folder + '/' + ode) as f:
            lines = f.readlines()

        x_vars_in_equations = []
        theta_vars = []

        for line in lines:
            cur_vars = preprocess_txt(line)
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

def selection_in_favor(theta_setups,
                       ode_systems,
                       target_folder,
                       models_psi,
                       Y,
                       T,
                       alpha,
                       z_half_alpha):
    theta_setups_idx = theta_setups['model'].unique()
    experiments_idx = theta_setups['experiment'].unique()

    if len(theta_setups_idx) != len(ode_systems):
        print('Exit on Error of theta_setups_idx != ODE_systems')
        sys.exit(1)

    models_est_params_by_exper = dict()

    for e in [0]:
        models_est_params = dict()

        for cur_system in range(len(ode_systems)):
            models_est_params[cur_system] = dict()

            cur_ODE_system = ode_systems[cur_system]
            model = int(cur_ODE_system.split('_')[-1].replace('.txt', ''))

            with open(target_folder + '/' + cur_ODE_system) as f:
                lines = f.readlines()

            cur_theta_setup = theta_setups[(theta_setups['model'] == model) &
                                           (theta_setups['experiment'] == e)]
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
                cur_vars = preprocess_txt(line)
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

            x_vars_dict = {x: x_replace(x) for x in x_vars}
            theta_vars_dict = {theta: theta_replace(theta) for theta in theta_vars}
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
            f = vertcat(*(eval(casadi_replace(ode, x_vars_dict, theta_vars_dict)) for ode in lines))
            dae = dict(x=x_symb, p=psi_symb, ode=f)

            V, H, Xhat = get_SW_HV(dae, T, xi_hat, psi_hat, sigma2_hat, Y, estimated, s)
            models_est_params[cur_system]['Xhat'] = Xhat
            models_est_params[cur_system]['Y'] = Y
            models_est_params[cur_system]['sigmas2_hat'] = sigma2_hat
            models_est_params[cur_system]['Vhat'] = V
            models_est_params[cur_system]['Hhat'] = H

        models_est_params_by_exper[e] = models_est_params

    model_combs = list(product(list(models_est_params.keys()), list(models_est_params.keys())))
    model_combs = list(set([tuple(sorted(x)) for x in model_combs if x[0] != x[1]]))

    model_selection_by_exper = []

    for e in models_est_params_by_exper.keys():
        models_est_params = models_est_params_by_exper[e]
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

            sw_value = sw_test_vec(X_hat_A, X_hat_B, sigmas2_A, sigmas2_B, Y_A, Y_B, V_A, H_A, V_B, H_B, h=None,
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

        model_selection_by_exper.append(model_selection)

    return pd.concat(model_selection_by_exper)