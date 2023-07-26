<a name="readme-top"></a>
<h3 align="center">Model Selection of ODE systems using Schennach-Wilhelm (S-W) test</h3>

  <p align="center">
    We develop innovative testing methodologies for ODE model selection in the presence of statistical noise. 
    While covering both the case of linear and nonlinear ODE systems, our approach builds upon a testing framework, 
    that sets it apart from the existing literature. In this repo, we provide Python code that allows to define ODE models, put data,
    specify estimating conditions, run estimation of parameters/initial values, and apply the S-W test for model selection.
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>See research paper »</strong></a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#foldercontents">Folder Contents</a></li>
      <ul>
        <li><a href="#data">Data</a></li>
        <li><a href="#models">Models</a></li>
        <li><a href="#data">Plots</a></li>
        <li><a href="#models">Scripts</a></li>
      </ul>
    <li><a href="#estimation">Estimation</a></li>
    <li><a href="#troubleshooting ">Troubleshooting </a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<a name="getting-started"></a>
<!-- GETTING STARTED -->
## Getting Started

We use Python 3.8 to run the code: you are free to try higher versions until stated in the requirements.txt file libraries can be installed. Download a local copy of the repo and run the model testing procedure via "run_model_selection.py" file by setting required options (see section "Usage").

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="usage"></a>
<!-- USAGE EXAMPLES -->
## Usage
In the root of the folder, there is a file, "run_model_selection.py", which can be used to estimate model parameters/initial values and to conduct model selection according to the model testing procedure described in the above paper. This script uses as input the data from the files "./data/estimation_setups.csv" and "./data/theta_setups.csv". The second one is the result of running an estimation procedure. So, in case you don't use it, you need to edit "./data/theta_setups.csv" by putting estimated/given values and defining appropriate options accordingly.

There are several options in the "SWtestModelSelection" module (that is in "run_model_selection.py"):
- "with_estimation" takes values "True/False": if you don't need estimation put "False", otherwise keep default "True";
- "alpha" is the level at which the test rejects the null hypothesis. The default is 0.05.
- "log_transform": if you need to bring your non-negative data to a similar scale, you can set this option to "True", which will apply the following transformation to all observations: $ln(Y+1)$, where adding 1 is needed to avoid taking logarithm over zeros;
- "B" is the number of primary initializations (see "Estimations" section for details) in the estimation procedure. The default is 1000. In the case when "with_estimation=False" it doesn't play any role.
 - "BB" is the number of secondary initializations in the estimation procedure. The default is 100. In the case when "with_estimation=False" it doesn't play any role.
 - "n_plot" is the number of time points that will be used in plotting each ODE system. Keep "None" if you want to plot the solutions for the original time vector, or put any number greater than the current number of observations to make curves smoother.

In the "SWtestModelSelection" module there is only one function "run()", that starts estimation and/or model selection procedures. The result of the last one is the file './data/model_selection_results.csv'.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="foldercontents"></a>
## Folder Contents
<a name="data"></a>

### Data
This folder contains 4 files: 'data.csv', 'estimation_setups.csv', 'theta_setups.csv', 'model_selection_results.csv'.

***'data.csv'***

This file contains observations:
- the first column should be always time (or any other variable with respect to which state derivates are calculated);
- other columns represent the observed states and should be in the same order as states in the model`s equations (the first column after time corresponds to the first equation in all models, the second column - to the second equation, etc.);
- **important**: in the case when not all states in ODE models are observed, you need to put all equations corresponding to the unobserved states after equations for observed states. For example, in the S.I.R. model, usually, only the "Infected" state is measured, thus epidemiological ODE systems should be defined in the way where the equation for "I" state is the first: "I.S.R", "I.S.E.R.", etc.
  
The names of columns could be anything, but they are used in plots: the first column name as a label for the x-axis, and others - as names for y-label in corresponding plots of these states.

***'estimation_setups.csv'***

This file defines what and how you want to estimate.
- **"model"** and **"parameter"** columns define the names of the state's $\sigma^2$ (in the form of "sigma1", "sigma2", ...), the state's initial values (in the form of "xi1", "xi2", ...) and parameters (in the form of "psi1", "psi2", etc.) for the corresponding model (1, 2, ..., K). Please note, while the notation is "sigma1", "sigma2",... you need to provide the corresponding variance $\sigma^2$ (if it known) and not standard deviation;
- put known values into the column **"given"**: they will not be estimated and used as it is to solve ODE systems. Also, they will be not used in $\hat{V}_n$ and $\hat{H}_n$ matrices. Leave this column empty if you want to estimate this creature;
- **"lower_bound"** and **"upper_bound"** columns are intended to specify possible lower and upper bounds for the corresponding value of the parameter. For "sigma1", "sigma2",... parameters keep these columns empty;
- in **"lower_initial_value"** and **"upper_initial_value"** you can provide the interval boundaries within which the initial guess for the parameter value will be chosen from the uniform distribution ($U\sim(li, bi)$) to run the minimization task. If you keep these columns blank, the initialization will be done from $U\sim(0,1)$, $U\sim(-1,0)$, and $U\sim(-1,1)$ depending on provided "lower/upper_bound". These columns should be both fulfilled with values or both kept empty. For "sigma1", "sigma2",... parameters keep these columns empty;

***'theta_setups.csv'***

This file is a result of running an optimization task to obtain MLE parameters. In the case of using our estimation procedure, this file is created automatically, so no need to edit it. If you want to use your MLE values, update 'theta_setups.csv' as follows:
- column **"theta"** should contain names of $\sigma^2$, initial values and model's parameters;
- column **"value"** should contain a value for the corresponding parameter;
- column **"model"** defines "k-th" number of the model to which the current value of the corresponding parameter belongs;
- column **"estimated"** defines if the current value was estimated (put 1) or given (put 0);
- as for the column **"to_include"**: put 1 for each 'psi1', 'psi2', ...; 1 for each 'sigma1', 'sigma2',... and 'xi1', 'xi2',... if the corresponding state is observed, otherwise put 0.

***model_selection_results.csv***

This file is a result of running model selection procedure:
- "Model A" and "Model B" columns define the pair of models (where numbers correspond to the "k-th" subscript in the names of ".txt" files and names of functions in "./script/models.py" file - see section "Models" for details);
- "sw_value" is the value of S-W test t-statistic, calculated according to the paper;
- "in favor" indicates which model is closer to the true data-generating process in the Kullback-Leibler divergence sense. "-" means that null hypotheses (that two models are equally distant from the DGP) cannot be rejected. The significance of rejecting is defined by the "alpha" parameter of the "SWtestModelSelection" module.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<a name="models"></a>

### Models
The folder "./models/" contains "ODE_system_k.txt" files, where k=1,2,...,K is the corresponding number of the model. In the file name "ODE_system_k.txt" only "k" should be edited. Each such file contains equations of an ODE system, where each equation should be placed in a new row. 

In the current repo, there are 4 files corresponding to 4 predator-prey models stated in the paper. Here is an example of how corresponding files are edited with the Lotka-Volterra model, which appears as model "1":

***Mathematical equations:***
```math
\left\{\begin{matrix}
x_1^{\prime}(t) = \psi_2\psi_3x_1(t)x_2(t)-\psi_4x_1(t) \hfill\\
x_2^{\prime}(t) =\psi_1x_2(t)-\psi_2x_1(t)x_2(t)\hfill
\end{matrix}\right.
```      
***Corresponding lines in "./data/ODE_system_1.txt":***
```
psi2 * psi3 * x2 * x1 - psi4 * x1
psi1 * x2 - psi2 * x2 * x1
```
***Corresponding function in "./scripts/models.py"***
```
def model1(x, t, psi1, psi2, psi3, psi4):
    x1, x2 = x
    d_x1 = psi2 * psi3 * x2 * x1 - psi4 * x1
    d_x2 = psi1 * x2 - psi2 * x2 * x1
    return np.array([d_x1, d_x2])
```
Thus for each model, you need to create a separate "./data/ODE_system_K.txt" file and add K functions to the "./scripts/models.py" file. The latter are used to solve a system of ordinary differential equations via "scipy.integrate.odeint" function (please, refer to the official documentation for the function's syntaxis).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<a name="plots"></a>

### Plots

In this folder, different plots are saved when running estimation (decline of the loss function versus the number of initializations) and model selection (observations and fitted models versus time) procedures.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<a name="scripts"></a>

### Scripts
This folder contains two files: "sw_test.py" and "models.py". In the first one, estimation and model selection algorithms are defined: "SWtestModelSelection" module is imported into "run_model_selection.py". The second one contains ODE system functions and should be edited according to the section "Models".

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<a name="estimation"></a>

## Estimation
To run MLE parameter estimation, set "with_estimation=True" in "SWtestModelSelection" module in "run_model_selection.py" file.

Parameters estimation  $\eta = (\xi, \psi)$ (whenever each is needed) for each model is obtained by using MLE estimator of the form:
```math
\hat{\eta}_n = \underset{\eta}{argmin}\sum_{j=1}^d\sum_{i=1}^n(Y_{ji}-x_j(t_i;\eta))^2
```
We implemented the SLSQP optimizer from "scipy" Python library to run minimization tasks.

To estimate parameters $\sigma^2$ (if needed) for each model the following MLE estimator is used:
```math
\hat{\sigma}_{j}^2 = \frac{1}{n}\sum_{i=1}^n(Y_{ji}-x_j(t_i;\eta))^2
```
Before launching estimations you need to define properly the required information for the optimization process in "./data/estimation_setups.csv" file (see section "Data").

The estimation procedure includes several steps: 
- run "B" number of estimations according to parameters defined in "estimation_setups.csv";
- save "B" number of loss functions and corresponding values of MLE parameters;
- choose MLE parameters with the lowest loss function;
- run "BB" number of estimations where initial values of parameters for minimization task is generated from $N(\hat{\theta}, \frac{1}{3}\sqrt{|\hat{\theta}|})$ ;
- choose MLE parameters with the lowest loss function;
- save chosen values to the "theta_setups.csv" file.

Thus:
- firstly, we optimize the loss function generating initial values of parameters in the set defined in "estimation_setups.csv";
- secondly, we optimize the loss function generating initial values of parameters in some neighborhood of MLE parameter values obtained in the first step.

Set BB=0 to skip the second step in estimations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<a name="troubleshooting"></a>

## Troubleshooting 
While running the model selection part of the code you can get the following error: 'CVodeB returned "CV_TOO_MUCH_WORK"'. It appears when the "casadi" package calculates the numerical derivatives for the current model (at which this error has arisen) and its corresponding MLE parameters. In this case, you need to check if the ODE system is defined properly and if the parameters make sense (sometimes it would require reestimating parameters with an increased "B" number of initializations or using another procedure).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<a name="contact"></a>

## Contact

Oleksandr Laskorunskyi - [@Linkedin](https://www.linkedin.com/in/alexander-laskorunskiy/) - ode.model.selection@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<a name="acknowledgments"></a>

## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>
