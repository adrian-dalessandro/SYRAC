from src.experiment import get_finetune_args, load_and_run_experiment, load_and_run_evaluation
from datetime import datetime as dt
"""
Wrapper to run the finetune experiment
"""
if __name__ == '__main__':
    args = get_finetune_args()
    experiment_path = args.experiment
    print("Running experiment at: " + experiment_path)
    print("Current time: " + dt.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print(args)
    print("Running training script")
    load_and_run_experiment(experiment_path, args)
    print("Running evaluation script")
    load_and_run_evaluation(experiment_path, args)