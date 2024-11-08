import os
import tvm
import argparse
from tvm import relax
import utils.model as mu
import utils.crypto as cu
import utils.rowhammer as ru
from tvm.ir import Array
from dotenv import load_dotenv

ROWHAMMER_ACCURACY_THRESHOLD = 0.25

def main():
    load_dotenv()
    secret_key = str(os.getenv("SECRET_KEY")).encode('ascii')

    # Receive input from user
    options = mu.OPTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help=f"Options: {', '.join(options.keys())}")
    args = parser.parse_args()

    # Validate
    if args.model not in options.keys():
        print("Model is not supported.")
        exit(-1)
    
    model, interactor, file_path, ex_t = options[args.model]

    target = 'llvm'
    device = tvm.cpu()

    # Import & lower the model
    model = mu.mu_import(model, interactor, file_path)
    model.eval()
    mod = mu.mu_export(model, ex_t)

    # Bind additional parameters)
    
    mod.show()

    mod, params = relax.frontend.detach_params(mod)
    params = params["main"]

    hashes = []
    hash_vars = []
    ctr = 0
    # Create hashes
    for i, param in enumerate(params):
        p = mod["main"].params[i+1]
        if 'b' not in p.name_hint: # Ignore biases for now
            name = "h" + str(ctr)
            ctr += 1
            h = (cu.cu_hash(param.asnumpy(), secret_key))
            hashes.append((name, h))
            hash_vars.append(relax.Var(name, relax.TensorStructInfo(h.shape, "int64")))



    params = [tvm.nd.array(param) for param in params]

    # Add hash parameters to the main function
    mod["main"] = relax.Function(list(mod["main"].params) + hash_vars, mod["main"].body)
    
    mod.show()

    # Perform model-specific transformations
    mod = interactor.transform(mod)

    mod, vm = mu.mu_build(mod, target, device)

    mod.show()

    # User interface
    while True:
        choice = input("Enter 't' to test the model, 'rh' to launch a Rowhammer attack, 'q' to quit: ")
        if choice == 't':
            accuracy = interactor.test(model, vm, params)
            print(f"Accuracy: {accuracy:.4f}")
        # Rowhammer attack until threshold is met
        elif choice == 'rh':
            print("Launching Rowhammer attack...")
            accuracy = 1
            while accuracy > ROWHAMMER_ACCURACY_THRESHOLD:
                for _ in range(10):
                    ru.ru_rowhammer(params)
                accuracy = interactor.test(model, vm, params)
                print(f"Accuracy: {accuracy:.4f}")
        # Quit
        elif choice == 'q':
            break
        else:
            pass

if __name__ == "__main__":
    main()
