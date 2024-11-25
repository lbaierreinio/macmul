from utils.timer import tu_get_runtime_probabilities
import utils.model as mu

def main():
    
    options = ['mlp1', 'mlp2', 'mlp3']
    probability_schedules = [[[0.8, 0.8, 0.8], [0.05, 0.05, 0.9]], [[0.8, 0.8, 0.8, 0.8, 0.8], [0.05, 0.05, 0.05, 0.05, 0.9]], [[0.8, 0.8, 0.8], [0.05, 0.05, 0.9]]]

    for i,o in enumerate(options):
        # Define model, hardware, target, and device
        model, interactor, file_path, ex_t, _, _ = mu.OPTIONS[o]

        tu_get_runtime_probabilities(model, interactor, file_path, ex_t, iterations_per_test=5000, budget=3, probability_schedules=probability_schedules[i], path=f"experiments/probabilities_{o}.txt")

if __name__ == "__main__":
    main()