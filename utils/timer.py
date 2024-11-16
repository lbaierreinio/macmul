import os
import tvm
import time
import argparse
import numpy as np
import utils.model as mu
import matplotlib.pyplot as plt

def tu_get_line(model, interactor, file_path, ex_t, *, iterations_per_budget, lo, hi, step, plot_path=None):
    budget_times = []
    budgets = []
    
    # Test out a number of different budgets
    for budget in range(0, 2000, 25):
        mod, vm, params, hs, ps = mu.mu_get_model_and_vm(model, interactor, file_path, ex_t, budget) # Build model with the given budget
        all_params = [*params, *hs, *ps]
        budget_t = []
        for i in range(0, iterations_per_budget):
            start = time.perf_counter()
            vm["main"](tvm.nd.array(ex_t), *all_params)[0]
            end = time.perf_counter()
            budget_t.append(end - start)
        
        budgets.append(budget)
        budget_times.append(np.mean(budget_t))
        print(f"{budget} {np.mean(budget_times)}")

    m, b = np.polyfit(budget_times[1:], budgets[1:], 1)

    if plot_path is not None:
        plt.plot(budgets, budget_times, marker='o')
        plt.xlabel('Number of Hashes')  # Label for the x-axis
        plt.ylabel('Budget Times')      # Label for the y-axis
        plt.title('Budget Times vs. Number of Hashes')
        plt.savefig(plot_path, format='pdf')
    
    return m, b

