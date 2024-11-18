from utils.timer import tu_get_detection_probabilities
import utils.model as mu

def main():
    options = ['mlp1', 'mlp2', 'mlp3']
    hashes = [3,5,3]
    for i,o in enumerate(options):
        # Define model, hardware, target, and device
        model, interactor, file_path, ex_t, _, _ = mu.OPTIONS[o]

        tu_get_detection_probabilities(model, interactor, file_path, ex_t, iterations_per_bit_flip=1000, budget_lo=hashes[i], budget_hi=hashes[i] + 1500, budget_step=500, bit_flip_lo=1, bit_flip_hi=11, bit_flip_step=1, plot_path=f"experiments/security_{o}.pdf")

if __name__ == "__main__":
    main()
