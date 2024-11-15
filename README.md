# DNN-MAC
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

### legacy
* Files that are no longer in use or were created solely for testing purposes.

> **Note:** The .so file needs to be readable and executable. chmod +rx /path/to/libmac_mul.so

Compile with gcc using this:
 g++ -shared -fPIC -o libmac_mul.so cpp/mac_mul.cpp -I/w/340/lucbr/tvm/include -I/w/340/lucbr/tvm/3rdparty/dmlc-core/include -I/w/340/lucbr/tvm/3rdparty/dlpack/include
