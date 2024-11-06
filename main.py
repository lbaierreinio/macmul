import tvm
import argparse
import utils.model as mu
import utils.crypto as cu
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
    
    model, interactor, file_path, ex_t = options[args.model]

    target = 'llvm'
    device = tvm.cpu()

    # Import & lower the model
    model = mu.mu_import(model, interactor, file_path)
    model.eval()
    mod = mu.mu_export(model, ex_t)

    # Perform model-specific transformations
    mod = interactor.transform(mod)
    
    mod.show()

    mod, vm, params = mu.mu_build(mod, target, device)

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
