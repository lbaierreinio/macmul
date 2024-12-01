from utils.timer import tu_get_degredation
import utils.model as mu

def main():
    # Define model, hardware, target, and device
    options = ['mlp1', 'mlp2', 'mlp3']

    for o in options:
        model, interactor, file_path, ex_t, _, _ = mu.OPTIONS[o]

        # Get the line of best fit for the inference_limit we want.
        tu_get_degredation(model, interactor, file_path, ex_t, iterations=5, step=10, stop_accuracy=0.25, stop_rowhammers=500, plot_path=f"experiments/degredation_{o}.pdf")

if __name__ == "__main__":
    main()