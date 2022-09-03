Poisson Generalized Additive Model (PGAM)
=====================================


A PGAM for the estimation neural tuning functions. Responses are rerpresented in terms of B-splines regularized by a smoothing enforcing penalization. B-spline coefficients and regularization hyperparameters are jointly learned from the data by numerical optimization of the a cross-validation score. The model infers maginal confidence bounds for the contribution of each feature to the neural response, and uses such bonuds to  identify the minimal subset of features each neuron reponds to.  See [[1]](#1) for a more details.

Table of Contents
=================
* [Setup](#setup)
     * [Conda environment](#conda-environment)
     * [Docker image](#docker-image)
* [Usage](#usage)
   * [Notebooks](#notebooks)
   * [Model parameters](#model-parameters)
* [References](#references)


# Setup
The package was implemented on macOS (Monterey version 12.5.1), and tested on Linux (RHEL version 8.4) and Windows 10. 
It requires python (>=3.6), and R (>= 3.6). 


Below we will provide two recomanded ways of setting up the package:

1. Install all software requirerments and create a dedicated [conda environment](#conda-environment)
2. Download and run a [Docker image](#docker-image).

#### Conda environemnt
1. Download  and install <a href="https://www.r-project.org/">R<a> and <a href="https://www.anaconda.com/products/distribution"> Anaconda <a>. 

2. Open the terminal/command prompt and create a conda environment with
	
	```sh
	conda create -n pgam python=3.9
	```

3. Activate the environment and install the required python packages

	```sh
	conda activate pgam
	conda install numpy pandas dill scikit-learn matplotlib -y
	conda install seaborn pyyaml h5py numba -y
	pip install rpy2 opt_einsum statsmodels
	```

4. Install the R package *survey*. The recommended option is to install the package directly through rpy2 with the following steps:
	<ol type="a", start="a">
  		<li>Run python, import r utils and install the package <em>survey</em>
  		</li>
  		</ol> 
  		```sh
  		python
  		from rpy2.robjects.packages import importr
  		utils = importr('utils')
  		utils.install_packages('survey')
		```
		<ol type="a", start="b">
  		<li>Select a mirror, proceed with the installation.</li>
  		<li>Exit python.
  		```sh
  		exit()</li>
  		```
	


#### Docker image

# Usage

## Demo

## Notebook



# References
<a id="1">[1]</a> 
<a href="https://proceedings.neurips.cc/paper/2020/hash/94d2a3c6dd19337f2511cdf8b4bf907e-Abstract.html">
Balzani, Edoardo , et al., 
"Efficient estimation of neural tuning during naturalistic behavior."
Advances in Neural Information Processing Systems 33 (2020): 12604-12614.<a>