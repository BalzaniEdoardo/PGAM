Poisson Generalized Additive Model (PGAM)
=====================================


A PGAM for the estimation neural tuning functions. Responses are rerpresented in terms of B-splines regularized by a smoothing enforcing penalization. B-spline coefficients and regularization hyperparameters are jointly learned from the data by numerical optimization of the a cross-validation score. The model infers maginal confidence bounds for the contribution of each feature to the neural response, and uses such bonuds to  identify the minimal subset of features each neuron reponds to.  See [[1]](#1) for a more details.

Table of Contents
=================
* [Setup](#setup)
     * [Conda environment](#conda-environment)
     	* [Inspect and edit the PATH environment variable](#env-var)
     * [Docker image](#docker-image)
     	* [Working with jupyter](#working-with-jupyter)
     	* [Running a script](#running-a-script)
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

## Conda environment

Instruction on how to set up a conda environment with the specified packages are provided below.

1. Download  and install <a href="https://www.r-project.org/">R<a> and <a href="https://www.anaconda.com/products/distribution"> Anaconda <a>. Make sure that the path to R is in the system PATH environment variable after the installation, [add it otherwise](#env-var). 

2. Open the terminal/command prompt and create a conda environment with
	
	```
	conda create -n pgam python=3.9
	```

3. Activate the environment and install the required python packages

	```
	conda activate pgam
	conda install jupyter
	conda install numpy pandas dill scikit-learn matplotlib -y
	conda install seaborn pyyaml h5py numba -y
	pip install rpy2 opt_einsum statsmodels
	```

4. Install the R package *survey*. The recommended option is to install the package directly through rpy2 with the following steps:
	<ol type="a"; start="a">
  		<li>Run python, import r utils and install the package <em>survey</em>
  		</li>
  	</ol> 
  	
  	```
	python
  	from rpy2.robjects.packages import importr
  	utils = importr('utils')
  	utils.chooseCRANmirror(ind=1) # any CRAN mirror id
  	utils.install_packages('survey')
  	exit()
	```


Test the installation by *cd* to the *PGAM/GAM_library* directory and run,

```
	python
	from GAM_library import *
	exit()
```

### Inspect and edit the PATH environment variable

On **windows**:
	
Inspect the PATH variable by entering in the command prompt,

```
	echo %PATH:;=&echo.%
```
 
The R home folder shold be listed  (usually *C:\R\R-version\\*).
 
 If not, edit the PATH variable (Windows 10):
 
 1. open the "Control Panel"
 
 2. search for "environment"
 
 3. click on "Edit the system environment variables"->"Edit variables..."
 
 4. scroll on "System variables" and click on "Path"
 
 5. click on "Edit..." -> "New", type the path to the R home folder and click "Ok".

 6. close the pop-up by clicking on "Ok"

 7. click "Apply" and then "Ok".

 8. restart the commad prompt.





<br><br>
On **mac OS X** and **Linux**,

Inspect the PATH variable by entering in the terrminal,
 
```
	echo $PATH | tr : '\n'
```
	

If the R home folder is not listed,

On **mac OS X**:

1. 	Open the .bash_profile file in your home directory (for example, /Users/your-user-name/.bash_profile) in a text editor.

2. 	Add <br>
	*export PATH="your-dir:$PATH"* <br>
	to the last line of the file, where your-dir is the R home directory.
	
3.	Save the .bash_profile file.

4.	Restart your terminal.
    
On **Linux**:

1.	Open the .bashrc file in your home directory (for example, /home/your-user-name/.bashrc) in a text editor.

2.	Add <br>
	*export PATH="your-dir:$PATH"* <br>
	 to the last line of the file, where your-dir is the R home directory.
	
3.	Save the .bashrc file.

4.	Restart your terminal.

  
## Docker image

Downloadinng a Docker image and running it in a Docker container is very simple and makes the setup of the package trivial. However, working with docker containers requires some familiarity with the docker syntax (starting, stopping and removing containers, mounting volumes etc.); I would recommand checking out one of the many tutorial available online before starting to work with docker. 

### Installing and running the PGAM Docker image


Download, install and start <a href="https://docs.docker.com/get-docker/"> Docker<a>. 

Download the PGAM Docker image with the terminal/command propmpt command

```
docker pull  edoardobalzani87/pgam:1.0
```

You can check the list of all the dowloaded images with the command,
```
docker images
```

You can run the image in a Docker container and inspect the contents with the command,

```
docker run -ti  edoardobalzani87/pgam:1.0 /bin/bash
```
 
The command will run a Linux bash shell that allows you to inspect the image content. python, R and all the required packages  are already installed and the enviironment variables are set up. Type ```exit```, to exit the bash shell and stop the container. 

You can delete a stopped container  with the command  ```docker rm CONTAINER-ID``` . The container ID can be found with the command ```docker ps -a```, which will list all available containers, their IDs, the image that they run and the command that they execute.

### Working with jupyter

Run the PGAM image in a container and launch jupyter notebook with the following command,

```
	docker run   -v your-notebook-folder:/notebooks -ti -p 8888:8888 edoardobalzani87/pgam:1.0
```

The -v option mounts the folder *your-notebook-folder*  of your coputer (the host computer) as a volume in the Docker container virtual file system linking it to the folder */notebook* .  

Files saved by the container in the */notebook* virtual folder will be automatically copied in *your-notebook-folder*, and files already present in *your-notebook-folder* will be automatically copied in */notebook* when the container is started. 

FIles that the container saves in other directories of the virtual file system will be lost once the container is stopped or removed (the container as a temporary file system).

The -p *local-port:contanier-port* option connects the port 8888 of the container with that of the host operating system, allowing the container and the operating system to interact.

Open a browser, and browse to *localhost:8888/* to connect to jupyter. You can test the library by working with the "PGAM Tutorial.ipynb" or you can crerate your own notebook. Files will be stoerd in the *your-notebook-folder*.

The ```run``` command  creates a new container each time, however, if you haven't removed an old contaiiner, it can be restarted with the command ```docker start CONTAINER-ID```. Inspect the inactive containers with ```docker ps -a```. You can stop a container with  ```docker stop CONTAINER-ID```

### Running a scripts

If you want to run *yourscript.py*  enter the code,

```
docker run -v your-script-folder/:/scripts -ti -p 8888:8888 edoardobalzani87/pgam:1.0 /bin/bash -c "python scripts/yourscript.py"
```

The -v option mounts *your-script-folder* as a volume in the contaiiner, links it to the virtual folder */scripts*, while the -c option executes a shell command, in this case *python yourscript.py*. 

Note that eventual the inputs loaded by *yourscript.py* needs to be saved in *your-script-folder* to be available within the container. Similarly, all the outputs that *yourscript.py* saves, must be saved in the virtual folder *scripts/* to be copied in the host file system.

# Usage

## Demo

## Notebook



# References
<a id="1">[1]</a> 
<a href="https://proceedings.neurips.cc/paper/2020/hash/94d2a3c6dd19337f2511cdf8b4bf907e-Abstract.html">
Balzani, Edoardo , et al., 
"Efficient estimation of neural tuning during naturalistic behavior."
Advances in Neural Information Processing Systems 33 (2020): 12604-12614.<a>
