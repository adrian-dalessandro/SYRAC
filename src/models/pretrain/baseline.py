from tensorflow.keras.layers import Concatenate, Subtract, Activation
import tensorflow as tf
from src.models.utils import get_encoder, get_rank_decoder
from typing import Tuple, Dict
from src.layers.preprocess import Preprocess

class BaselineRank(tf.keras.Model):
    """
    A siamese image ranking model, which outputs a pair of continuous values,
    before applying a subtraction layer to get the difference between the two values.
    This can be used for training models for pairwise ranking. 
    Args:
        model_name (str): either 'vgg16' or 'resnet50'
        input_shape (tuple): the shape of the input image
        weights (str): the weights to use (either 'imagenet' or None)
        initializations (str): the initializations to use
        num_neurons (int): the number of neurons in the dense layer
        num_layers (int): the number of dense layers 
        augmentor (tf.keras.layers.Layer): a keras layer that augments the input image
    """
    def __init__(self, model_name: str, input_shape: Tuple[int, ...], weights: str, 
                 initializations: str, num_neurons: int, num_layers: int, augmentor: tf.keras.layers.Layer):
        super(BaselineRank, self).__init__()
        # Load the encoder model
        self.encoder = get_encoder(model_name, input_shape, weights)
        # Load the decoder models
        self.rank_decoder = get_rank_decoder(self.encoder, initializations, num_neurons, num_layers)
        # Store the augmentor
        self.augmentor = augmentor
        self.subtract = Subtract(name="rank")
        self.activation = Activation("sigmoid")
        self.concat = Concatenate(axis=0, name="concatenate")
        self.preprocess = Preprocess("resnet50")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        # Augment the input data
        aug_imgs = [self.preprocess(self.augmentor(img, training=training)) for img in inputs]
        img_ab = tf.stop_gradient(self.concat(aug_imgs))
        # Get the output of the feature encoder
        z_ab = self.encoder(img_ab, training=training)
        z_a, z_b = tf.split(z_ab, 2, axis=0)
        # Get the ranking prediction
        rank_pred_logits = self.subtract([self.rank_decoder(z_a, training=training),
                                          self.rank_decoder(z_b, training=training)])
        output = self.activation(rank_pred_logits)
        return output
    
    def compile(self, 
                optimizer: tf.keras.optimizers.Optimizer, 
                loss: tf.keras.losses.Loss):
        super(BaselineRank, self).compile()
        # Store the optimizer and loss function
        self.optimizer = optimizer
        self.loss = loss

        # define metrics to track during training
        self.rank_accuracy = tf.keras.metrics.BinaryAccuracy(name="rank_accuracy")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.rank_accuracy]

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        inputs, y = data
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.rank_accuracy.update_state(y, y_pred)
        self.total_loss_tracker.update_state(loss)
        return {"loss": self.total_loss_tracker.result(), "rank_accuracy": self.rank_accuracy.result()}

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        inputs, y_rank = data

        # synthetic ranking evaluation step
        y_pred = self(inputs, training=False)
        loss = self.loss(y_rank, y_pred)
        self.rank_accuracy.update_state(y_rank, y_pred)
        self.total_loss_tracker.update_state(loss)
        return {"loss": self.total_loss_tracker.result(),
                "rank_accuracy": self.rank_accuracy.result()}
    
    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """
        Overwrite the save method with a custom method which savs ONLY the encoder model
        """
        self.encoder.save(filepath, overwrite=overwrite, save_format=save_format, **kwargs)
