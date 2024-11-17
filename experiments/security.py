from utils.timer import tu_get_probability_of_detection
import utils.model as mu

def main():
    # Define model, hardware, target, and device
    model, interactor, file_path, ex_t, _, _ = mu.OPTIONS['mlp']

    tu_get_probability_of_detection(model, interactor, file_path, ex_t, iterations_per_bit_flip=10000, budget_lo=10, budget_hi=20, budget_step=10, bit_flip_lo=1, bit_flip_hi=5, bit_flip_step=1, plot_path="experiments/security.pdf")

if __name__ == "__main__":
    main()
