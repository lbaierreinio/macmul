#!/bin/bash

pushd ~

source /w/340/lucbr/miniconda3/bin/activate
export TVM_HOME=/w/340/lucbr/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
conda activate tvm-build-venv

popd