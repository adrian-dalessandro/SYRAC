import argparse
import json
from datetime import datetime as dt
from pathlib import Path
import os
import importlib.util
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

"""
Setup scripts for loading all of the parameters for a single experiment. This includes a function to
load the command-line arguments
"""
def get_args():
    """
    loads the following command-line arguments:
    1. index: the index of the experiment to run
    2. dataset: the dataset to use
    3. data_dir: the directory containing the dataset
    4. experiment_dir: the directory to save the experiment to
    5. experiment_name: the name of the experiment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, help='Path to the experiment directory')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--experiment_dir', type=str, default='experiments')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--params_dir', type=str)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument("--filter_classes", nargs="*", default=[], help="List of classes to filter out")
    args = parser.parse_args()
    return args

def get_finetune_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, help='Path to the decoder directory')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--N', type=int)
    parser.add_argument('--experiment_path', type=str, default='experiments')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model to finetune')
    parser.add_argument('--params_dir', type=str)
    return parser.parse_args()

def setup(args, hyparams):
    # print all args from get_args in one line
    print("Arguments:")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # print all hyparams from get_hyperparams in one line
    print("Hyparams:")
    print(' '.join(f'{k}={v}' for k, v in hyparams.items()))

    exp_dir = args.experiment_dir + "/" + args.experiment_name + "/" + args.dataset + "/" + str(args.index) + "/" + dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # Save the hyperparameters to a json file
    with open(exp_dir + '/hyperparams.json', 'w') as f:
        json.dump(hyparams, f)
    with open(exp_dir + "/args.json", 'w') as f:
        json.dump(vars(args), f)
    return exp_dir

def get_callbacks(exp_dir, monitor_1, patience):
    return [ModelCheckpoint(filepath=exp_dir + '/best_model_{epoch:02d}', monitor=monitor_1, 
                            save_best_only=False, save_format='tf', verbose=1, mode='min'),
            EarlyStopping(monitor=monitor_1, patience=patience),
            CSVLogger(exp_dir + '/results.csv')]

def load_and_run_experiment(experiment_path, args):
    train_script_path = os.path.join(experiment_path, 'train.py')
    
    # Load training module
    spec = importlib.util.spec_from_file_location("train", train_script_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    # Run training with arguments
    train_module.train(args)

def load_and_run_evaluation(experiment_path, args):
    eval_script_path = os.path.join(experiment_path, 'eval.py')
    
    # Load training module
    spec = importlib.util.spec_from_file_location("eval", eval_script_path)
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    
    # Run training with arguments
    eval_module.eval(args)

def augment_config(factor):
    zoom_args = {"height_factor": (0, 0.60/factor), "fill_mode": "constant"}
    blur_args = {"kernel_size": 4, "sigma": 1/factor}
    color_args = {"brightness": 0.4/factor, "jitter": 0.15/factor}
    contrast_args = {"factor": 0.6/factor}
    noise_args = {"intensity": 0.1/factor}
    model_args = {"model_name": "resnet50"}
    config = {"zoom_args": zoom_args, "blur_args": blur_args, "color_args": color_args,
              "contrast_args": contrast_args, "noise_args": noise_args, "model_args": model_args}
    return config
