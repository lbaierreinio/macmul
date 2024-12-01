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
* You can also change the protection parameters of a model that you want to test within the OPTIONS variable. For example, you can change the number of chunks or the probability schedule. which are the last two variables in each entry of the OPTIONS list.
* The behavior of this system on non-Pytorch models with layers not in [ReLU, Linear] is undefined. 

### utils
* A set of modules with various helper functions.

### experiments
* A set of Python files, which when run, execute, and save the results of their experiments to a pdf or txt file. You may modify the experiment files to change the set of models that they act on or the parameters of the experiment.

### main.py
* USAGE: `python3 ./main.py {model_name}`. The list of available model names can be found in the OPTIONS variable of utils/model.py. Currently, you can select one of 'mlp1', 'mlp2', or 'mlp3'. These will add MACMul protection to the specified model. Then, you will be able to enter the following commands into the user interface for fast experiments:
* st: Test a single prediction
* srh: Perform a single Rowhammer attack
* t: Get accuracy on the entire validation accuracy
* rh: Perform Rowhammer attacks until validation accuracy drops below a specified ROWHAMMER_ACCURACY_THRESHOLD. (Note that if the model is protected, the program will exit long before this ROWHAMMER_THRESHOLD_ACCURACY is reached).
* q: Quit

## Running Experiments
The experiments can be found under the experiments directory. Each experiment will produce a series of .txt or .pdf files, one for each MLP that the experiment is run on. Each .py file in the experiments directory corresponds to one experiment. It defines the MLPs that the experiments will be run on, and iterates over each MLP, calling a function in utils/timer.py that holds the underlying logic of that experiment. None of these experiments require arguments from the command line. If you’d like to change the parameters of the experiments, you can do so within the Python script directory. The experiments will overwrite the results of the previous experiments. Note that some experiments may take at least a few minutes to run, or more, if you modify the parameters of the experiment.

More detail about each experiment is provided below:

### runtime.py
This experiment analyzes the average runtime of each MLP as the number of chunks to secure that MLP increases. The results of these experiments will be saved as degredation_{model_name}.pdf.

### security.py
This experiment analyzes the likelihood of detecting N bit flips for MLPs secured by ~5, ~500, and ~1000 chunks. Each layer is assigned a 100% probability of tag verification. 1000 inferences are run for each value of N to compute the likelihood. The results of these experiments will be saved as security_{model_name}.pdf.

### degradation.py
This experiment analyzes the effectiveness of random bit-flips to degrade the accuracy of the model. It continues flipping random bits until the accuracy is degraded below 25%. Models are trained and validated on the MNIST dataset for this experiment. The results of these experiments will be saved as degredation_{model_name}.pdf

### probabilities.py
This experiment analyzes the average runtime of MLPs protected by different probability schedules, given a fixed number of chunks (1 per layer). The results of these experiments will be saved as probabilities_{model_name}.txdt


## Additional Notes
* When I import Apache TVM, the terminal displays the following error message: 'Error: Using lLVM ...'. This is expected and does not interfere with the results of the experiments or program.
* During the security.py experiment, you may periodically see the following error messages in the console: RuntimeWarning: Overflow encountered in multiply. Again, this is fine and does not interfere with the results of the experiments. It is simply a byproduct of the bit-flips.
