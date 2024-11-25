import os
import tvm
import time
import argparse
import numpy as np
import utils.model as mu
import utils.rowhammer as ru
import matplotlib.pyplot as plt

def tu_get_line(model, interactor, file_path, ex_t, *, iterations_per_budget, lo, hi, step, plot_path=None):
    budget_times = []
    budgets = []
    
    # Test out a number of different budgets
    for budget in range(lo, hi, step):
        mod, vm, params, hs, ps, prs = mu.mu_get_model_and_vm(model, interactor, file_path, ex_t, budget) # Build model with the given budget
        all_params = [*params, *hs, *ps, *prs]
        budget_t = []
        for i in range(0, iterations_per_budget):
            start = time.perf_counter()
            vm["main"](tvm.nd.array(ex_t), *all_params)[0]
            end = time.perf_counter()
            budget_t.append(end - start)
        
        budgets.append(budget)
        time_in_ns = np.mean(budget_t) * 1e3
        budget_times.append(time_in_ns)
        print(f"Number of Chunks: {budget} Average Run-time: {round(time_in_ns,3)} (ms)")

    m, b = np.polyfit(budget_times[1:], budgets[1:], 1)

    if plot_path is not None:
        plt.plot(budgets, budget_times, marker='o')
        plt.xlabel('Number of Chunks')  # Label for the x-axis
        plt.ylabel('Run-time (ms)')      # Label for the y-axis
        plt.title('Run-time vs. Number of Chunks')
        plt.savefig(plot_path, format='pdf')
        plt.close()
    
    return m, b

def tu_get_detection_probabilities(
        model, 
        interactor, 
        file_path, 
        ex_t, 
        *,
        iterations_per_bit_flip, 
        budget_lo, 
        budget_hi, 
        budget_step,
        bit_flip_lo,
        bit_flip_hi,
        bit_flip_step, 
        plot_path
    ):

    # Try out different number of hashes
    all_budgets = []
    all_probabilities = []
    num_bit_flips = [i for i in range(bit_flip_lo, bit_flip_hi, bit_flip_step)]
    # Each budget will correspond to a different line
    for budget in range(budget_lo, budget_hi, budget_step):
        mod, vm, params, hs, ps, prs = mu.mu_get_model_and_vm(model, interactor, file_path, ex_t, budget)
        ground_truth_params = [p.copyto(tvm.cpu()) for p in params]
        probabilities = []    
        # Iterate over different bit flips
        for num_bit_flip in range(bit_flip_lo, bit_flip_hi, bit_flip_step):
            # Sample iterations_per_bit_flip times
            num_bit_flips_detected = 0
            for _ in range(0, iterations_per_bit_flip):
                # Perform this number of bit flips
                for _ in range(0, num_bit_flip):
                    params = ru.ru_rowhammer(params)

                all_params = [*params, *hs, *ps, *prs]

                # Run the model
                try: 
                    vm["main"](tvm.nd.array(ex_t), *all_params)[0]
                except:
                    num_bit_flips_detected += 1
                params = [p.copyto(tvm.cpu()) for p in ground_truth_params]
            probabilities.append(num_bit_flips_detected / iterations_per_bit_flip)
            print(f"{budget} chunks detected {num_bit_flip} bit flips with probability {num_bit_flips_detected / iterations_per_bit_flip}")
        # Save probabilities for this budget
        all_probabilities.append(probabilities)
        all_budgets.append(budget)
    
    # Plot the probabilities
    for i, (b, p) in enumerate(zip(all_budgets, all_probabilities)):
        plt.plot(num_bit_flips, p, label=f'{b} Chunks', marker='o')  # Use x_axis_values

    # Add labels, title, and legend
    plt.xlabel('Number of Bit Flips')
    plt.ylabel('Probability of Bit Flip Detection')
    plt.title('Bit Flip Detection')
    plt.legend(title='Number of Chunks')
    plt.grid(True)
    plt.savefig(plot_path, format='pdf')
    plt.close()

def tu_get_runtime_probabilities(model, interactor, file_path, ex_t, *, iterations_per_test, budget, probability_schedules, path):
    f = open(path, "w")
    probability_schedules.insert(0, [0.] * len(probability_schedules[0]))
    for probability_schedule in probability_schedules:
        mod, vm, params, hs, ps, prs = mu.mu_get_model_and_vm(model, interactor, file_path, ex_t, budget, probability_schedule)
        all_params = [*params, *hs, *ps, *prs]
        times = []
        for i in range(0, iterations_per_test):
            start = time.perf_counter()
            vm["main"](tvm.nd.array(ex_t), *all_params)[0]
            end = time.perf_counter()
            times.append(end - start)
        
        f.write("---------------------SCHEDULE---------------------\n")
        for i, p in enumerate(probability_schedule):
            f.write(f"Layer {i}: Probability {p}\n")
        
        f.write(f"Average Run-time: {round(np.mean(times) * 1e3, 3)} (ms)\n\n")
    f.close()

def tu_get_degredation(model, interactor, file_path, ex_t, *, iterations, stop_accuracy, step, plot_path):
    
    accuracies = []
    num_rowhammers = []

    for _ in range(0, iterations):
        mod, vm, params, hs, ps, prs = mu.mu_get_model_and_vm(model, interactor, file_path, ex_t, 0)
        prs = [tvm.nd.array([0.]) for _ in prs]
        all_params = [*params, *hs, *ps, *prs]

        a = []
        num_rowhammer = []
        
        accuracy = interactor.test(model, vm, all_params)
        a.append(accuracy)
        num_rowhammer.append(0)
        while accuracy >= stop_accuracy:
            for _ in range(0, step):
                params = ru.ru_rowhammer(params)
            all_params = [*params, *hs, *ps, *prs]
            accuracy = interactor.test(model, vm, all_params)
            a.append(accuracy)
            num_rowhammer.append(num_rowhammer[-1] + step)

        accuracies.append(a)
        num_rowhammers.append(num_rowhammer)

    # Plot the accuracies each as separate lines
    for i, (a, n) in enumerate(zip(accuracies, num_rowhammers)):
        plt.plot(n, a, label=f'Run {i}', marker='o')  # Use x_axis_values

    plt.xlabel('Number of Bit Flips')
    plt.ylabel('Accuracy')
    plt.title('Degredation of Model Accuracy')
    plt.grid(True)
    plt.savefig(plot_path, format='pdf')
    plt.close()

