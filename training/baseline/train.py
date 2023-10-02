from src.experiment import setup, get_callbacks, augment_config
from src.models.pretrain.baseline import BaselineRank
from src.data.loader import RankDataIterator
from src.layers.augment import AugmentModel
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json

def train(args):
    hyparams = json.load(open(args.params_dir + "/baseline.params.json"))
    aug_config = augment_config(hyparams['augment'])
    exp_dir = setup(args, hyparams)

    # Open a json file containing the dataset information
    # using the dataset name and data_dir from args
    with open(args.data_dir + "/" + args.dataset + "/annotations.json", 'r') as f:
        dataset = json.load(f)
        
    # Load the dataset
    classes_to_filter = set(args.filter_classes)
    train_data = list(filter(lambda x: x["info"]["type"] not in classes_to_filter, dataset["train"]))
    val_data = list(filter(lambda x: x["info"]["type"] not in classes_to_filter, dataset["val"]))
    
    # Load the dataset
    data_iter = RankDataIterator(train_data, 
                                 args.data_dir, 
                                 hyparams["input_shape"])
    data_iter = data_iter.build(hyparams["batch_size"], True, True)
    val_iter = RankDataIterator(val_data, 
                                args.data_dir, 
                                hyparams["input_shape"]) # The validation dataset iterator
    val_iter = val_iter.build(1, False, False)

    # Load the model
    augmentor = AugmentModel(**aug_config)
    model = BaselineRank(augmentor= augmentor, model_name=hyparams['model_name'],
                            input_shape=hyparams['input_shape'],
                            weights=hyparams['weights'], initializations=hyparams['initializations'], 
                            num_neurons=hyparams['num_neurons'], num_layers=hyparams['num_layers'])

    # Compile the model
    optimizer = Adam(learning_rate=hyparams['learning_rate'])
    model.compile(optimizer=optimizer, loss=tf.keras.losses.get(hyparams['loss']))

    # Creates callbacks for saving the best model, early stopping, and logging results
    callbacks = get_callbacks(exp_dir, 'val_loss', args.patience)

    # Train the model
    model.fit(data_iter, epochs=hyparams['epochs'], callbacks=callbacks, 
            validation_data=val_iter, validation_freq=1)