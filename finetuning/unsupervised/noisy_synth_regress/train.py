import tensorflow as tf
from sklearn import linear_model
from src.finetune.utils import _train_setup_utils
import joblib
"""
NOISY REGRESSION TRAINING
---------------------------------
"""
def train(args):
    dir_name = "noisy_synth_regression"
    features, counts, path_name, _ = _train_setup_utils(args, dir_name)

    # Load the linear model
    reg = linear_model.LinearRegression(positive=True, fit_intercept=False)

    # Reshape the features and counts
    features = tf.reshape(features, [-1, 2048])
    counts = tf.reshape(counts, [-1, 1])

    # Fit the linear model
    reg.fit(features, counts)

    # Save the linear model
    joblib.dump(reg, path_name + "/linear.joblib")
