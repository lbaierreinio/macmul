from utils.timer import tu_get_detection_probabilities
import utils.model as mu

def main():
    options = ['mlp1', 'mlp2', 'mlp3']

    for o in options:
        # Define model, hardware, target, and device
        model, interactor, file_path, ex_t, _, _ = mu.OPTIONS[o]

        tu_get_detection_probabilities(model, interactor, file_path, ex_t, iterations_per_bit_flip=10000, budget_lo=10, budget_hi=20, budget_step=10, bit_flip_lo=1, bit_flip_hi=5, bit_flip_step=1, plot_path=f"experiments/security_{o}.pdf")

if __name__ == "__main__":
    main()
