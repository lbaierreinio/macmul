import os
import tvm
import argparse
import utils.model as mu
import utils.helpers as hp
import utils.rowhammer as ru

ROWHAMMER_ACCURACY_THRESHOLD = 0.25

def main():
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
    target = 'llvm'
    device = tvm.cpu()

    model = mu.mu_import(model, interactor, file_path) # Import model from PyTorch or train if not found
    model.eval() # Set to evaluation mode
    mod = mu.mu_export(model, ex_t) # Export model to IRModule
    mod, params = mu.mu_detach_params(mod) # Detach parameters
    mod, hs = mu.mu_integrate_hashes(mod, params, hp.get_secret_key()) # Integrate hashes into main function
    mod = interactor.transform(mod) # Transform IRModule
    mod, vm = mu.mu_build(mod, target, device) # Build for our target & device

    mod.show()
    while True:
        choice = input("Enter choice (st, srh, t, rh, q): ")
        try: 
            if choice == 'st': # Single Test
                interactor.test(model, vm, [*params, *hs], True)
            elif choice == 'srh': # Single Rowhammer
                ru.ru_rowhammer(params)
            elif choice == 't': # Bulk Test on Accuracy
                accuracy = interactor.test(model, vm, [*params, *hs])
                print(f"Accuracy: {accuracy:.4f}")
            elif choice == 'rh': # Rowhammer until certain threshold is met
                print("Launching Rowhammer attack...")
                accuracy = 1
                while accuracy > ROWHAMMER_ACCURACY_THRESHOLD:
                    for _ in range(10):
                        ru.ru_rowhammer(params)
                    accuracy = interactor.test(model, vm, [*params, *hs])
                    print(f"Accuracy: {accuracy:.4f}")
            elif choice == 'q': # Quit
                break
            else:
                pass
        except AssertionError:
            print("Weights have been tampered with. Exiting...")
            exit()

if __name__ == "__main__":
    main()
