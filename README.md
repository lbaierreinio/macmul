# MACMul
The purpose of this project is to use the Apache TVM framework to implement MAC integrity protection in DNNs. This is important as Rowhammer attacks are becoming more feasible with increasing DRAM density. Attacks have shown that intelligently flipping bits a small number of bits within DNN parameters can significantly degrade the performance of the model.

## Steps to Run
ssh lucbr@compos0.cs.toronto.edu

cd into final project directory

source init.sh

python3 ./main.py [model]

## Codebase Structure
### models
* PyTorch model definitions and interactor classes to train and test the models.

### utils
* A set of modules with various helper functions.

### experiments
* A set of Python files, which when run, execute, and save the results of their experiments to a pdf or txt file.

### main.py
When executed, adds protection to the DNN as specified by the parameters provided by the user, and then enters a user interface where the user can run a series on the model (e.g. perform a Rowhammer attack, test accuracy).
