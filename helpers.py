import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import argparse, os, sys
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('svg')

#-----------------------------networks----------------------------------

def define_tiny_cnn(num_classes, input_shape):
  #define model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=input_shape, padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))


  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512,activation="relu"))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.BatchNormalization())
    
  model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

  #compile model
  opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model

def define_small_cnn(num_classes, input_shape):
  #define model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(filters=256, kernel_size = (3,3), activation="relu", input_shape=input_shape, padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

  model.add(tf.keras.layers.Conv2D(filters=512, kernel_size = (3,3), activation="relu", padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    
  model.add(tf.keras.layers.Conv2D(filters=512, kernel_size = (3,3), activation="relu", padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(1024,activation="relu"))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.BatchNormalization())
    
  model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

  #compile model
  opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
  return model

def define_base_cnn(num_classes, input_shape):
  #define model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(filters=256, kernel_size = (3,3), activation="relu", input_shape=input_shape, padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
  model.add(tf.keras.layers.Conv2D(filters=256, kernel_size = (3,3), activation="relu", padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.2))

  model.add(tf.keras.layers.Conv2D(filters=512, kernel_size = (3,3), activation="relu", padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
  model.add(tf.keras.layers.Conv2D(filters=512, kernel_size = (3,3), activation="relu", padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.2))
    
  model.add(tf.keras.layers.Conv2D(filters=512, kernel_size = (3,3), activation="relu", padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
  model.add(tf.keras.layers.Conv2D(filters=512, kernel_size = (3,3), activation="relu", padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.2))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(1024,activation="relu"))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.BatchNormalization())
    
  model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

  #compile model
  opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model

def define_transformer(transformer_layers, mlp_head_units, x_train, num_classes, input_shape):
  #define hyperparameters
  weight_decay = 0.0001
  learning_rate = 0.001
  image_size = 72  # We'll resize input images to this size
  patch_size = 6  # Size of the patches to be extracted from the input images
  num_patches = (image_size // patch_size) ** 2
  projection_dim = 64
  num_heads = 4
  transformer_units = [
    projection_dim * 2,
    projection_dim,
  ]  # Size of the transformer layers
  mlp_head_units = mlp_head_units  # Size of the dense layers of the final classifier

  #do some data preprocessing
  data_augmentation = keras.Sequential(
  [
      tf.keras.layers.experimental.preprocessing.Normalization(),
      tf.keras.layers.experimental.preprocessing.Resizing(image_size, image_size),
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
      tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
      tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
  ],
  name="data_augmentation",
  )
  # Compute the mean and the variance of the training data for normalization.
  data_augmentation.layers[0].adapt(x_train)

  #set input layer/shape
  inputs = layers.Input(shape=input_shape)

  # Augment data
  augmented = data_augmentation(inputs)

  # Create patches
  patches = Patches(patch_size)(augmented)

  # Encode patches
  encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

  # Create multiple layers of the Transformer block.
  for _ in range(transformer_layers):
      # Layer normalization 1.
      x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
      attention_output = layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=projection_dim, dropout=0.1
      )(x1, x1)
      # Skip connection 1.
      x2 = layers.Add()([attention_output, encoded_patches])
      # Layer normalization 2.
      x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
      # MLP.
      x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
      # Skip connection 2.
      encoded_patches = layers.Add()([x3, x2])

  # Final normalization/output
  representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
  representation = layers.Flatten()(representation)
  representation = layers.Dropout(0.5)(representation)

  #add mlp to transformer
  features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

  #pass features from mlp to final dense layer/classification
  logits = layers.Dense(num_classes)(features)

  #create model
  model = keras.Model(inputs=inputs, outputs=logits)

  optimizer = tfa.optimizers.AdamW(
      learning_rate=learning_rate, weight_decay=weight_decay
  )

  model.compile(
      optimizer=optimizer,
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[
          keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
          keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
      ],
  )

  return model

def create_models(data_set, network_size, input_shape, x_train):
  if data_set == "cifar10" or data_set == "fashion_mnist":
    num_classes = 10
  elif data_set == "cifar100":
    num_classes = 100
    
  if network_size == "tiny":
    cnn_model = define_tiny_cnn(num_classes, input_shape)
    vit_model = define_transformer(4, [512, 1024], x_train, num_classes, input_shape)
  elif network_size == "small":
    cnn_model = define_small_cnn(num_classes, input_shape)
    vit_model = define_transformer(6, [1024, 2048], x_train, num_classes, input_shape)
  elif network_size == "base":
    cnn_model = define_base_cnn(num_classes, input_shape)
    vit_model = define_transformer(8, [1024, 2048], x_train, num_classes, input_shape)
    
  cnn_model.summary()
  vit_model.summary()
  
  return cnn_model, vit_model

#-----------------------------transformer classes and methods----------------------------------
#multilayer perceptron
def mlp(x, hidden_units):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        return {"patch_size": self.patch_size}

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim=projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def get_config(self):
        return {"num_patches": self.num_patches, "projection_dim": self.projection_dim}

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim=projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def get_config(self):
        return {"num_patches": self.num_patches, "projection_dim": self.projection_dim}

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

#-----------------------------training----------------------------------

#train CNN of defined size (tiny, small, base)
def train_cnn(cnn_model, x_train, y_train):
  cnn_history = cnn_model.fit(x_train, y_train, epochs=15, batch_size=256, validation_split=0.2)
  return cnn_history

#train vision transformer of defined size (tiny, small, base)
def train_vit(vit_model, x_train, y_train):
  vit_history = vit_model.fit(x_train, y_train, epochs=15, batch_size=256, validation_split=0.2)
  return vit_history

#-----------------------------testing----------------------------------

def test_models(cnn_model, vit_model, x_test, y_test):
  print("Making predictions on test data...")
  print("CNN metrics: ")
  predict_metrics(cnn_model, x_test, y_test)

  print("Vision Transformer metrics: ")
  predict_metrics(vit_model, x_test, y_test)

def predict_metrics(model, x_test, y_test):
  #predict and format output to use with sklearn
  predict = model.predict(x_test)
  predict = np.argmax(predict, axis=1)
  #macro precision, recall, and F1 score
  precision_macro = precision_score(y_test, predict, average='macro')
  recall_macro = recall_score(y_test, predict, average='macro')
  f1_macro = f1_score(y_test, predict, average='macro')
  #micro precision, recall, and F1 score
  precision_micro = precision_score(y_test, predict, average='micro')
  recall_micro = recall_score(y_test, predict, average='micro')
  f1_micro = f1_score(y_test, predict, average='micro')

  print("Macro precision: ", precision_macro)
  print("Micro precision: ", precision_micro)
  print("Macro recall: ", recall_macro)
  print("Micro recall: ", recall_micro)
  print("Macro F1 score: ", f1_macro)
  print("Micro F1 score: ", f1_micro)

def plot_results(history, data_set, network_size, model):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title(f'Model accuracy on {data_set} dataset using {network_size} sized {model} network')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.savefig(f'./figures/accuracy_{model}_{data_set}_{network_size}')
  plt.clf()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title(f'Model loss on {data_set} dataset using {network_size} sized {model} network')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.savefig(f'./figures/loss_{model}_{data_set}_{network_size}') 
  plt.clf()
  print("Plots saved")

#-----------------------------data handling----------------------------------

def get_data(data_set):
  print("Loading data...")

  if data_set == "cifar10":
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  elif data_set == "cifar100":
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
  elif data_set == "fashion_mnist":
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

  #normalize
  x_train = x_train / 255
  x_test = x_test / 255

  return (x_train, y_train), (x_test, y_test)

def save_models(cnn_model, vit_model, data_set, network_size):
  cnn_model.save_weights(f"./models/{data_set}_{network_size}_CNN.h5")
  vit_model.save_weights(f"./models/{data_set}_{network_size}_vision_transformer.h5")
  print("Models saved.")

def load_models(cnn_model, vit_model, data_set, network_size):
  print("Loading models...")
  #load model weights
  cnn_model.load_weights(f"./models/{data_set}_{network_size}_CNN.h5")
  vit_model.load_weights(f"./models/{data_set}_{network_size}_vision_transformer.h5")
  #initialize optimizers
  cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  vit_optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
  #compile CNN model
  cnn_model.compile(
    optimizer=cnn_optimizer, 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])
  #compile ViT model
  vit_model.compile(
    optimizer=vit_optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
      keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
      keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
  )

  return cnn_model, vit_model

#-----------------------------argument handling----------------------------------

def check_data_set(user_input):
  error_msg = ("%s is an invalid argument (must be either \"cifar10\","
                "\"cifar100\"), or \"fashion_mnist\"" % user_input)
  valid_input = ['cifar10', 'cifar100', 'fashion_mnist']

  for val in valid_input:
    if user_input == val:
      return user_input
  raise argparse.ArgumentTypeError(error_msg)

def check_network_size(user_input):
  error_msg = ("%s is an invalid argument (must be either \"tiny\","
                "\"small\"), or \"base\"" % user_input)
  valid_input = ['tiny', 'small', 'base']

  for val in valid_input:
    if user_input == val:
      return user_input
  raise argparse.ArgumentTypeError(error_msg)

def get_args():
  # Initiate the parser
  parser = argparse.ArgumentParser()

  # Add arguments
  parser.add_argument('network_size', type=check_network_size)
  parser.add_argument('data_set', type=check_data_set)

  # Read arguments from the command line
  args = parser.parse_args()
  return args