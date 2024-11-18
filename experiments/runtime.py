from utils.timer import tu_get_line
import utils.model as mu

def main():
    # Define model, hardware, target, and device
    options = ['mlp1', 'mlp2', 'mlp3']

    for o in options:
        model, interactor, file_path, ex_t, _, _ = mu.OPTIONS[o]

        # Get the line of best fit for the inference_limit we want.
        tu_get_line(model, interactor, file_path, ex_t, iterations_per_budget=10, lo=0, hi=2000, step=25, plot_path=f"experiments/runtime_{o}.pdf")

if __name__ == "__main__":
    main()