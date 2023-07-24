![image](https://github.com/odemodelselection/sw_test/assets/139265720/92645926-d3a0-4709-9318-23dfc4d6a9fc)<h3 align="center">Model Selection of ODE systems using S-W test</h3>

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
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->
## Getting Started

Download a local copy of the repo and run the model testing procedure with the following simple steps. We use Python 3.8 to run the code: you are free to try higher versions until stated in the requirements.txt file libraries can be installed.

<!-- USAGE EXAMPLES -->
## Usage
In the root of the folder, there are two main files: "run_parameter_estimation.py" and "run_model_selection.py". The first one can be used to estimate model parameters/initial values based on the options defined in the "./data/estimation_setups.csv" file. The second one is used to conduct model selection according to the model testing procedure described in the above paper. This file uses as input the data from the file "./data/theta_setups.csv" which is the resulting output of running "run_parameter_estimation.py". So, in case you don't use our estimation procedure, you need to edit "./data/theta_setups.csv" by putting estimated/given values and defining appropriate options accordingly.

### Data
This folder contains 3 files: 'data.csv', 'estimation_setups.csv', and 'theta_setups.csv'.

***'data.csv'***

This file contains observations:
- the first column should be always time (or any other variable with respect to which state derivates are calculated);
- other columns represent the observed states and should be in the same order as states in the model`s equations (the first column after time corresponds to the first equation in all models, the second column - to the second equation, etc.);
- **important**: in the case when not all states in ODE models are observed, you need to put all equations corresponding to the unobserved states after equations for observed states. For example, in the S.I.R. model, usually, only the "Infected" state is measured, thus epidemiological ODE systems should be defined in the way where the equation for "I" state is the first: "I.S.R", "I.S.E.R.", etc.
  
The names of columns could be anything, but they are used in plots: the first column name as a label for the x-axis, and others - as names for y-label in corresponding plots of these states.

***'estimation_setups.csv'***

This file defines what and how you want to estimate.
- **"model"** and **"parameter"** columns define a state $\sigma^2$ (in the form of "sigma1", "sigma2", ...), a state initial value (in the form of "xi1", "xi2", ...), parameter (in the form of "psi1", "psi2", etc.) of the corresponding model for which you are going to define values in the rest of the columns. Please note, while the notation is "sigma1", "sigma2",... you need to provide the corresponding variance $\sigma^2$ (if it known) and not standard deviation;
- put known values into the column **"given"**: they will not be estimated and used as it is to solve ODE systems. Also, they will be not used in $\hat{V}_n$ and $\hat{H}_n$ matrices. Leave this column empty if you want to estimate this creature;
- **"lower_bound"** and **"upper_bound"** columns are intended to specify possible lower and upper bounds for the corresponding value of the parameter. For "sigma1", "sigma2",... parameters keep these columns empty;
- in **"lower_initial_value"** and **"upper_initial_value"** you can provide the interval boundaries within which the initial guess for the parameter value will be chosen from the uniform distribution ($U\sim(li, bi)$) to run the minimization task. If you keep these columns blank, the initialization will be done from $U\sim(0,1)$, $U\sim(-1,0)$, and $U\sim(-1,1)$ depending on provided "lower/upper_bound". These columns should be both fulfilled with values or both kept empty. For "sigma1", "sigma2",... parameters keep these columns empty;

***'theta_setups.csv'***

This file is a result of running an optimization task to obtain MLE parameters. In the case of using our estimation procedure, this file is created automatically, so no need to edit it. If you want to use your MLE values, update 'theta_setups.csv' as follows:
- column **"theta"** should contain names of $\sigma^2$, initial values and model's parameters;
- column **"value"** should contain a value for the corresponding parameter;
- column **"model"** defines "k-th" number of the model to which the current value of the corresponding parameter belongs;
- column **"estimated"** defines if the current value was estimated (put 1) or given (put 0);
- as for the column **"to_include"**: fill with 1 for each 'psi1', 'psi2', ...; fill with 1 for each 'sigma1', 'sigma2',... and 'xi1', 'xi2',... if the corresponding state was observed, otherwise put 0.

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

### Plots

The folder where different plots are saved through estimation runs (decline of the loss function versus the number of initializations) and model selection runs (observations and fitted models versus time).

### Scripts

### Estimation
Parameters estimation  $\eta = (\xi, \psi)$ (whenever each is needed) for each model is obtained by using MLE estimator of the form:
```math
\hat{\eta}_n = \underset{\eta}{argmin}\sum_{j=1}^d\sum_{i=1}^n(Y_{ji}-x_j(t_i;\eta))^2
```
We implemented the SLSQP optimizer from "scipy" Python library to run minimization tasks.

To estimate parameters $\sigma^2$ (if needed) for each model the following MLE estimator is used:
```math
\hat{\sigma}_{j}^2 = \frac{1}{n}\sum_{i=1}^n(Y_{ji}-x_j(t_i;\eta))^2
```
Before launching "run_parameter_estimation.py" file you need properly define in "./data/estimation_setups.csv" required for the optimization process information:


<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
