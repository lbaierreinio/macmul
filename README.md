# MACMul
MACMul is an integrity protection system that uses Apache TVM & CMAC-AES to detect tampered weights in neural networks.

## Requirements:
- Conda
- Apache TVM

## Steps to Run
Update init.sh to point to your Apache TVM installation and the name of your conda environment (you can create your conda environment from the environment.yml file). 

source init.sh

From here you have two options:

a) Enter the user interface via python3 ./main.py [model-name] (test models are defined under the /models directly).
b) Run the experiments in the experiments directory.

## Codebase Structure
### models
* PyTorch model definitions and interactor classes to train and test the models.
* If you'd like to create a new model for testing, you can do so under this directory. Note that you will also have to register your model in the OPTIONS variable in helpers/model to use it in the user interface or experiments.
* The behavior of this system on non-Pytorch models with layers not in [ReLU, Linear] is undefined. 

### utils
* A set of modules with various helper functions.

### experiments
* A set of Python files, which when run, execute, and save the results of their experiments to a pdf or txt file.

### main.py
* When executed, adds protection to the DNN as specified by the parameters provided by the user, and then enters a user interface where the user can run a series of commands on the model. The commands available are listed below:
* st: Single Test
* srh: Single Rowhammer
* t: Test on entire validation set
* rh: Perform Rowhammer attacks until validation accuracy drops below a specified ROWHAMMER_ACCURACY_THRESHOLD.
* q: Quit
