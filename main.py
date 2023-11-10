import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
#import hydra
from moviepy.editor import *
import logging

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from utils import create_dataset
#from omegaconf import DictConfig, OmegaConf
from model import create_LRCN_model

logger = logging.getLogger(__name__)

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

#@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main():
    #logger.info(cfg.params)
    #features, labels, video_files_paths = create_dataset()
    # Construct the required LRCN model.
    LRCN_model = create_LRCN_model()
    # Display the success message.
    print("Model Created Successfully!")

    features, labels, video_files_paths = create_dataset()
    one_hot_encoded_labels = to_categorical(labels)
    # Split the Data into Train ( 75% ) and Test Set ( 25% ).
    features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                            test_size = 0.25, shuffle = True,
                                                                            random_state = seed_constant)
    #plot_model(LRCN_model, to_file = 'LRCN_model_structure_plot.png', show_shapes = True, show_layer_names = True)

    # Create an Instance of Early Stopping Callback.
    early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)
 
    # Compile the model and specify loss function, optimizer and metrics to the model.
    LRCN_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

    # Start training the model.
    LRCN_model_training_history = LRCN_model.fit(x = features_train, y = labels_train, epochs = 70, batch_size = 4 ,shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])


    return

if __name__ == "__main__":
    main()