from utils.timer import tu_get_runtime_probabilities
import utils.model as mu

def main():
    # Define model, hardware, target, and device
    model, interactor, file_path, ex_t, _, _ = mu.OPTIONS['mlp']

    # Get the line of best fit for the inference_limit we want.
    probability_schedules = [[0.8, 0.8, 0.8], [0.05, 0.05, 0.9]]
    tu_get_runtime_probabilities(model, interactor, file_path, ex_t, iterations_per_test=5000, budget=3, probability_schedules=probability_schedules)

if __name__ == "__main__":
    main()