import numpy as np
from tqdm import tqdm
import tensorflow as tf
import json
from src.data.loader import CountDataIterator
from src.layers.preprocess import Preprocess
from sklearn.metrics import r2_score
from src.visualization import error_rate_boxplot, visualize_patch_count
from src.data.images import patchify
from datetime import datetime
from pathlib import Path

def get_features_utils(data_iter, model, prepare):
    # Extract the features from the pre-trained model
    features = []
    counts = []
    for batch in tqdm(data_iter):
        images, labels = batch
        images = prepare(images)
        features.append(model(images).numpy())
        counts.append(labels.numpy())

    features = np.concatenate(features, axis=0)
    counts = np.concatenate(counts)
    return features, counts

def model_loader_utils(args):
    print("Loading model at: {}".format(args.experiment_path + "/" + args.model_path))
    model = tf.keras.models.load_model(args.experiment_path + "/" + args.model_path)
    prepare = Preprocess("resnet50")
    return model, prepare

def data_loader_utils(args, hyparams, batch_size, dataset, key=None):
    with open(args.data_dir + "/" + dataset + "/annotations.json", 'r') as f:
        if key is not None:
            data = json.load(f)[key]
        else:
            data = json.load(f)
        
    synth_iter = CountDataIterator(data, args.data_dir, hyparams["input_shape"])
    synth_iter = synth_iter.build(batch_size, False, False)
    return synth_iter

def get_patch_features(data_iter, model, prepare, N):
    features = []
    counts = []
    for batch in tqdm(data_iter):
        images, labels = batch
        patches = tf.stack(patchify(images[0], N), axis=0)
        patched_image = prepare(patches)
        patch_feats = model(patched_image).numpy()
        features.append(patch_feats)
        counts.append(labels.numpy())

    features = np.array(features)
    counts = np.array(counts)
    return features, counts
    
def _eval_setup_utils(args, dir_name):
    path_name = args.experiment_path + "/" + args.model_path + "/" + dir_name
    # make the directory if it doesn't exist
    Path(path_name).mkdir(parents=True, exist_ok=True)
    # Load the hyperparameters
    hyparams = json.load(open(args.params_dir + "/baseline.params.json"))

    # Load the pre-trained model from the path. Must be stored in SavedModel format.
    model, prepare = model_loader_utils(args)

    # Load Dataset
    synth_iter = data_loader_utils(args, hyparams, 1, args.test_data, "test")

    # Extract the patch features from the pre-trained model
    features, gt_counts = get_patch_features(synth_iter, model, prepare, args.N)
    return features, gt_counts, path_name, synth_iter

def _train_setup_utils(args, dir_name):
    path_name = args.experiment_path + "/" + args.model_path + "/" + dir_name
    # make the directory if it doesn't exist
    Path(path_name).mkdir(parents=True, exist_ok=True)
    # Load the hyperparameters
    hyparams = json.load(open(args.params_dir + "/baseline.params.json"))

    # Load the pre-trained model from the path. Must be stored in SavedModel format.
    model, prepare = model_loader_utils(args)

    # Load Dataset
    synth_iter = data_loader_utils(args, hyparams, 3, args.train_data)

    # Extract the features from the pre-trained model
    features, counts = get_features_utils(synth_iter, model, prepare)
    return features, counts, path_name, synth_iter

def _eval_operations(args, predictions, gt_counts, path_name, synth_iter):
    gt_counts = gt_counts.reshape(-1)
    pred_counts = np.array(predictions).sum(axis=-1)

    # Calculate the mae, mse, and R^2
    mae = np.mean(np.abs(gt_counts - pred_counts))
    mse = np.mean(np.square(gt_counts - pred_counts))
    r2 = r2_score(gt_counts, pred_counts)

    # Save the information to a json file
    array_to_save = {"predictions": pred_counts.tolist(),
                    "counts": gt_counts.tolist(),
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                    "N": args.N}
    
    dtstr = datetime.now().strftime("%d%m%Y%H%M%S")
    json.dump(array_to_save, open(path_name + "/results.{}.json".format(dtstr), 'w'))
    # Visualize the results
    error_rate_boxplot(pred_counts, gt_counts, save_fig=True, fig_name = path_name + "/error_rate_boxplot.{}.png".format(dtstr))

    # visualize a few images annotated with their predicted counts
    for i, (x, y) in enumerate(synth_iter.take(10)):
        visualize_patch_count(x, args.N, predictions[i], y.numpy()[0], save_fig=True, fig_name=path_name + "/{}.png".format(i))