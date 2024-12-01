# MACMul
MACMul is an integrity protection system that uses Apache TVM & AES-CMAC to detect tampered weights in neural networks.

## Requirements:
- Conda
- Apache TVM

## Steps to Reproduce
I ran all of my experiments on the University of Toronto comps server, which I can SSH into through `[username]@comps2.cs.toronto.edu`.
The main requirements to run the code are Conda (Miniconda) and Apache TVM.
- Install Miniconda using the Linux installation described in this link: https://docs.anaconda.com/miniconda/install/.  
- Install Apache TVM using the steps described in this link: https://tvm.apache.org/docs/install/index.html. You may try installation via pip or Docker, but I did not try either of these methods. I installed Apache TVM from source. Instructions to install TVM from source are provided in the link above.
- Once Conda is installed, you need to run its activation script, depending on where you installed it. I installed it in the comps2 server under my w/340 directory:

`source /w/340/lucbr/miniconda3/bin/activate`

- I used Conda to install the necessary dependencies to run the project. The environment.yml file corresponding to the Conda environment I used is in the repository. You can create your Conda environment based on the environment.yml file: env create -f environment.yml.
Note that you may want to change the name & prefix lines in the environment.yml file before you create your Conda environment. The name is self-explanatory, and the prefix specifies where the environment will be created. Depending on where you are running this project, you may be able to leave the prefix blank, in which case Conda will use its default environment directory.
- If you installed Apache TVM from source, you will need to modify the PYTHONPATH environment variable to include the Python package directory for TVM, so that the MACMul project can locate it. I downloaded mine under my w/340 directory, so I used the following commands:

`export TVM_HOME=/w/340/lucbr/tvm`

`export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH`

- If you are also running this on the University of Toronto comps server, you may actually be able to point to my Apache TVM installation from your directory, which would save you from having to install it yourself. I haven’t verified that this works on other accounts, but my work directory should be open.

- Activate the Conda environment: 
`conda activate [environment-name].`

- Note that there is an init.sh file in this repository, which is a script containing the instructions I would run before using the project on the University of Toronto servers. You may also refer to this. Once you have followed these steps, you should be able to run experiments or enter the user interface. 


## Codebase Structure
### models
* PyTorch model definitions and interactor classes to train and test the models.
* If you'd like to create a new model for testing, you can do so under this directory. Note that you will also have to register your model in the OPTIONS variable in helpers/model to use it in the user interface or experiments.
* The behavior of this system on non-Pytorch models with layers not in [ReLU, Linear] is undefined. 

### utils
* A set of modules with various helper functions.

### experiments
* A set of Python files, which when run, execute, and save the results of their experiments to a pdf or txt file. You may modify the experiment files to change the set of models that they act on or the parameters of the experiment.

### main.py
* When executed, adds protection to the DNN as specified by the parameters provided by the user, and then enters a user interface where the user can run a series of commands on the model. The commands available are listed below:
* st: Single Test
* srh: Single Rowhammer
* t: Test on entire validation set
* rh: Perform Rowhammer attacks until validation accuracy drops below a specified ROWHAMMER_ACCURACY_THRESHOLD.
* q: Quit

## Running Experiments
The experiments can be found under the experiments directory. Each experiment will produce a series of .txt or .pdf files, one for each MLP that the experiment is run on. Each .py file in the experiments directory corresponds to one experiment. It defines the MLPs that the experiments will be run on, and iterates over each MLP, calling a function in utils/timer.py that holds the underlying logic of that experiment. None of these experiments require arguments from the command line. If you’d like to change the parameters of the experiments, you can do so within the Python script directory. Finally, I recommend that you remove the existing .pdf or .txt files corresponding to the last round of experiments before running a new set of experiments. 

More detail about each experiment is provided below:

### runtime.py
This experiment analyzes the average runtime of each MLP as the number of chunks to secure that MLP increases. The results of these experiments will be saved as degredation_{model_name}.pdf.

### security.py
This experiment analyzes the likelihood of detecting N bit flips for MLPs secured by ~5, ~500, and ~1000 chunks. Each layer is assigned a 100% probability of tag verification. 1000 inferences are run for each value of N to compute the likelihood. The results of these experiments will be saved as security_{model_name}.pdf.

### degradation.py
This experiment analyzes the effectiveness of random bit-flips to degrade the accuracy of the model. It continues flipping random bits until the accuracy is degraded below 25%. Models are trained and validated on the MNIST dataset for this experiment. The results of these experiments will be saved as degredation_{model_name}.pdf

### probabilities.py
This experiment analyzes the average runtime of MLPs protected by different probability schedules, given a fixed number of chunks (1 per layer). The results of these experiments will be saved as probabilities_{model_name}.txdt

