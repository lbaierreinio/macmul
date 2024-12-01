import os
import tvm
import argparse
import utils.timer as tu
import utils.model as mu
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
    model, interactor, file_path, ex_t, inference_limit, probabilities = options[args.model]

    # Get the line of best fit for the inference_limit we want.
    m, b = tu.tu_get_line(model, interactor, file_path, ex_t, iterations_per_budget=10, lo=0, hi=300, step=25)

    budget = int(m * inference_limit + b)

    mod, vm, params, hs, ps, prs = mu.mu_get_model_and_vm(model, interactor, file_path, ex_t, budget, probabilities)

    mod.show()
    all_params = [*params, *hs, *ps, *prs]
    while True:
        choice = input("Enter choice (st, srh, t, rh, q): ")
        try: 
            if choice == 'st': # Single Test
                interactor.test(model, vm, all_params, True)
            elif choice == 'srh': # Single Rowhammer
                ru.ru_rowhammer(params)
            elif choice == 't': # Bulk Test on Accuracy
                accuracy = interactor.test(model, vm, all_params)
                print(f"Accuracy: {accuracy:.4f}")
            elif choice == 'rh': # Rowhammer until certain threshold is met
                print("Launching Rowhammer attack...")
                accuracy = 1
                total = 0
                while accuracy > ROWHAMMER_ACCURACY_THRESHOLD:
                    for _ in range(10):
                        ru.ru_rowhammer(params)
                    total += 10
                    accuracy = interactor.test(model, vm, all_params)
                    print(f"Total Rowhammer attacks: {total}")
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
