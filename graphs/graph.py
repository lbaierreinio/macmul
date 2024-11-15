import os
import tvm
import argparse
import utils.model as mu

def retrieve_observations(mod, vm, params, hs):
    # Receive input from user
    options = mu.OPTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help=f"Options: {', '.join(options.keys())}")
    args = parser.parse_args()

    # Validate
    if args.model not in options.keys():
        print("Model is not supported.")
        exit(-1)

    
    # Define model, hardware, target, and device
    model, interactor, file_path, ex_t = options[args.model]

    mod, vm, params, hs = mu.mu_get_model_and_vm(model, interactor, file_path, ex_t)

    mod.show()

    # Iterate over number of hashes

    # Iterate over number of tests for this number of hashes
