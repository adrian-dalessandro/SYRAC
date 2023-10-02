from src.finetune.utils import _eval_operations, _eval_setup_utils
import joblib
import tensorflow as tf

"""
NOISY REGRESSION EVALUATION
---------------------------------

    Evaluate a regression count decoder on a real test dataset.

    This function loads a pre-trained feature encoder, a pre-trained regression model and a test dataset. It then makes predictions using the model,
    calculates evaluation metrics (MAE, MSE, R^2), and saves results and visualizations.

    Args:
        args (object): Configuration parameters and experiment details.

    Returns:
        None: Results and visualizations are saved to the specified directory.
"""
def eval(args):
    dir_name = "noisy_synth_regression"
    features, gt_counts, path_name, synth_iter = _eval_setup_utils(args, dir_name)

    # Load the linear regression model
    reg = joblib.load(path_name + "/linear.joblib")

    # Iterate over features, and patch features
    predictions = []
    for feature in features:
        tally = []
        for patch_feature in feature:
            patch_feature = tf.reshape(patch_feature, [-1, 2048])
            # Calculate the probability of each count category
            pred_count = reg.predict(patch_feature)
            tally.append(pred_count[0,0])
        predictions.append(tally)

    _eval_operations(args, predictions, gt_counts, path_name, synth_iter)
