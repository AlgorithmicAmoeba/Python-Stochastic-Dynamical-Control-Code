[![Build Status](https://travis-ci.org/darren-roos/Python-Stochastic-Dynamical-Control-Code.svg?branch=master)](https://travis-ci.org/darren-roos/Python-Stochastic-Dynamical-Control-Code)

# Install instructions

First prepare your environment

## 64-bit Ubuntu 16.04 LTS

Install [Anaconda with python 3.6](https://conda.io/docs/user-guide/install/index.html) (preferably 64-bit version, if possible)

Create a dedicated conda environment:

    conda create -n stochastic-models python=3.6 matplotlib=2.1.0 numpy=1.13.3 scipy=0.19.1 pandas=0.22.0 pathlib=1.0.1
    source activate stochastic-models
    conda install -c cvxgrp cvxpy=0.4.9 libgcc=7.2.0
    conda install -c mosek mosek=8.1.34

Install pyCharm community version from [their website](https://www.jetbrains.com/pycharm/download/#section=linux)
OR use the snap tool: 

    sudo snap install pycharm-community --classic

Run the following in a terminal:

    sudo apt-get install dvipng texlive-latex-extra

install git: 

    sudo apt-get install git 
    
## All environments

Clone the repo: 

    git clone https://github.com/darren-roos/Python-Stochastic-Dynamical-Control-Code.git

In PyCharm, set the python environment to the one create above:
1. Open the Python-Stochastic-Dynamical-Control-Code project in PyCharm
2. Go to the Project Structure menu (Shortcut: Ctrl+Alt+S)
3. Open the Project Interpreter Tab
4. Click the small cog icon and select Add Local
5. Select Conda Environment and then select existing environment
6. Locate the stochastic-models environment (should be in the /envs directory of your Anaconda install path)
