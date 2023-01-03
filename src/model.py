"""

> Plant Classification - Artificial Neural Networks and Deep Learning Project
> Politecnico di Milano - AY 2022/23
> Authors: Alberto Boffi, Francesco Bleggi

"""

import tensorflow as tf
import numpy as np
import os
import random

from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

tfk = tf.keras
tfkl = tf.keras.layers

# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

__batch_size = 32

def createFoldersAndCallbacks(model_name):
  
    exps_dir = os.path.join("/content/drive/MyDrive/model")
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)
      
    now = datetime.now().strftime('%b%d_%H-%M-%S')

    exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
      
    callbacks = []

    # Model checkpoint
    ckpt_dir = os.path.join(exp_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = os.path.join(ckpt_dir, 'cp.ckpt'), 
        save_weights_only = True, 
        save_best_only = False
    ) 
    callbacks.append(ckpt_callback)

    # Visualizing Learning on Tensorboard
    tb_dir = os.path.join(exp_dir, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
      
    # Losses and metrics for both training and validation
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir = tb_dir, 
        profile_batch = 0,
        histogram_freq = 1) 
    callbacks.append(tb_callback)

    # Early Stopping
    es_callback = tfk.callbacks.EarlyStopping(
        monitor = 'val_accuracy',
        mode = 'max',
        patience = 25,
        restore_best_weights = True
    )
    callbacks.append(es_callback)

    return callbacks

def displayConfusionMatrix(model, validation_set):

    y_pred_temp = model.predict_generator(
        validation_set,
        535 // __batch_size + 1
    )
    y_pred = np.argmax(
        y_pred_temp,
        axis = 1
    )
    target_names = ['1', '2', '3', '4', '5', '6', '7', '8']

    print("Confusion Matrix")
    print(confusion_matrix(
        validation_set.classes,
        y_pred
    ))
    print("Classification Report")
    print(classification_report(
        validation_set.classes,
        y_pred,
        target_names = target_names
    ))

def main():

    #######################################
    ########## Data Preparation ###########
    #######################################

    training_set_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip = True,
        vertical_flip = True,
        height_shift_range = 16,
        width_shift_range = 16,
        fill_mode = 'wrap'
    )

    validation_set_gen = tf.keras.preprocessing.image.ImageDataGenerator()

    training_set = training_set_gen.flow_from_directory(
        directory = r'/content/drive/MyDrive/train',
        target_size = (96, 96), 
        color_mode = 'rgb',
        class_mode = 'categorical',
        batch_size = __batch_size,
        shuffle = True,
        seed = seed
    )

    validation_set = validation_set_gen.flow_from_directory(
        directory = r'/content/drive/MyDrive/val',
        target_size = (96, 96), 
        color_mode = 'rgb',
        class_mode = 'categorical',
        batch_size = __batch_size,
        shuffle = False,
        seed = seed
    )

    #######################################
    ########## Transfer Learning ##########
    #######################################

    supernet = tfk.applications.ResNet50(
        include_top = False,
        weights = 'imagenet',
        input_shape = (96, 96, 3)
    )

    supernet.trainable = False

    # Network structure
    input_layer = tfk.Input(shape = (96, 96, 3))
    input_layer = supernet(input_layer)
    input_layer = tfkl.Flatten(name = 'Flattening')(input_layer)

    dp_layer_1 = tfkl.Dropout(
        0.3,
        seed = seed
    )(input_layer)

    dense_layer_1 = tfkl.Dense(
        128, 
        activation = 'relu',
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(dp_layer_1)

    dp_layer_2 = tfkl.Dropout(
        0.3,
        seed = seed
    )(dense_layer_1)

    dense_layer_2 = tfkl.Dense(
        64, 
        activation = 'relu',
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(dp_layer_2)

    dp_layer_3 = tfkl.Dropout(
        0.3,
        seed = seed
    )(dense_layer_2)

    output_layer = tfkl.Dense(
        8, 
        activation = 'softmax',
        kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(dp_layer_3)

    tl_model = tfk.Model(
        inputs = input_layer,
        outputs = output_layer,
        name = 'model'
    )

    # Model compilation
    tl_model.compile(
        loss = tfk.losses.CategoricalCrossentropy(),
        optimizer = tfk.optimizers.Adam(),
        metrics = 'accuracy'
    )
    tl_model.summary()

    # Training
    tl_history = tl_model.fit(
        x = training_set,
        batch_size = __batch_size,
        epochs = 500,
        validation_data = validation_set,
        callbacks = [tfk.callbacks.EarlyStopping(
            monitor = 'val_accuracy',
            mode = 'max',
            patience = 30,
            restore_best_weights = True
        )]
    ).history

    tl_model.save("/content/drive/MyDrive/model")
    
    displayConfusionMatrix(tl_model, validation_set)

    #######################################
    ############# Fine Tuning #############
    #######################################

    # Reloading the model after transfer learning
    ft_model = tfk.models.load_model("/content/drive/MyDrive/model")
    ft_model.summary()

    ft_model.load_weights("/content/drive/MyDrive/cp.ckpt")
    ft_model.get_layer('resnet50').trainable = True

    # Freezing the first n layers
    for i, layer in enumerate(ft_model.get_layer('resnet50').layers[:81]):
        layer.trainable=False

    for i, layer in enumerate(ft_model.get_layer('resnet50').layers):
        print(i, layer.name, layer.trainable)

    ft_model.summary()

    # Model compilation
    ft_model.compile(
        loss = tfk.losses.CategoricalCrossentropy(),
        optimizer = tfk.optimizers.Adam(5e-5),
        metrics = 'accuracy'
    )

    # Fine-tuning the model
    hom1_callbacks = createFoldersAndCallbacks(model_name = 'ft_model')

    # Training
    ft_history = ft_model.fit(
        x = training_set,
        batch_size = 32,
        epochs = 500,
        validation_data = validation_set,
        callbacks = hom1_callbacks
    ).history

    displayConfusionMatrix(ft_model, validation_set)

if __name__ == '__main__':
    main()