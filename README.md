# CSC2231 Final Project
## Steps to run on comps server

ssh lucbr@comps2.cs.toronto.edu

source /w/340/lucbr/miniconda3/bin/activate

export TVM_HOME=/w/340/lucbr/tvm

export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

conda activate tvm-build-venv

python3 ./main.py
