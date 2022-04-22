import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import argparse, os, sys
from sklearn.metrics import precision_score, recall_score, f1_score

#-----------------------------networks----------------------------------

def create_tiny_cnn(num_classes, input_shape):
  #define model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512, activation='relu'))
  model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
  #compile model
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  return model

def create_small_cnn(num_classes, input_shape):
  #define model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512, activation='relu'))
  model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
  #compile model
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  return model

def create_base_cnn(num_classes, input_shape):
  #define model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512, activation='relu'))
  model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
  #compile model
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  return model

#TODO: make_tranformer uses variables not present in scope of sampleViT.py example
def make_transformer(transformer_layers, num_classes, input_shape, x_train):
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
  inputs = layers.Input(shape=shape)

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
  model.summary()
  return model

#-----------------------------training----------------------------------
#each train function below trains CNN and vision transformer of defined size (tiny, small, base)

def train_tiny(num_classes, input_shape, x_train, y_train, x_test, y_test):
  #train tiny CNN
  cnn_model = create_tiny_cnn(num_classes, input_shape)
  history = cnn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
  #save tiny CNN model
  cnn_model.save("./models/tiny_CNN.h5")
  print("tiny_CNN model saved.")

  #train tiny vision transformer
  #transformer_layers = 4
  #vit_classifier = make_transformer(transformer_layers, num_classes, input_shape, x_train)

def train_small(num_classes, input_shape, x_train, y_train, x_test, y_test):
  #train small CNN
  cnn_model = create_small_cnn(num_classes, input_shape)
  history = cnn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
  #save small CNN model
  cnn_model.save("./models/small_CNN.h5")
  print("small_CNN model saved.")

  #train small vision transformer
  #transformer_layers = 6
  #vit_classifier = make_transformer(transformer_layers, num_classes, input_shape, x_train)

def train_base(num_classes, input_shape, x_train, y_train, x_test, y_test):
  #train base CNN
  cnn_model = create_base_cnn(num_classes, input_shape)
  history = cnn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
  #save base CNN model
  cnn_model.save("./models/base_CNN.h5")
  print("base_CNN model saved.")

  #train base vision transformer
  #transformer_layers = 8
  #vit_classifier = make_transformer(transformer_layers, num_classes, input_shape, x_train)

#-----------------------------testing----------------------------------

def test_tiny(x_test, y_test):
  print("Loading model")
  model = tf.keras.models.load_model("./models/tiny_CNN.h5")
  print("Making predictions on test data")
  predict_metrics(model, x_test, y_test)

def test_small(x_test, y_test):
  print("Loading model")
  model = tf.keras.models.load_model("./models/small_CNN.h5")
  print("Making predictions on test data")
  predict_metrics(model, x_test, y_test)

def test_base(x_test, y_test):
  print("Loading model")
  model = tf.keras.models.load_model("./models/base_CNN.h5")
  print("Making predictions on test data")
  predict_metrics(model, x_test, y_test)

def predict_metrics(model, x_test, y_test):
  #predict and format output to use with sklearn
  predict = model.predict(x_test)
  predict = np.argmax(predict, axis=1)
  y_test = np.argmax(y_test, axis=1)
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

#-----------------------------data handling----------------------------------

def get_data(data_set):
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

  #one-hot encode
  y_train = tf.keras.utils.to_categorical(y_train)
  y_test = tf.keras.utils.to_categorical(y_test)

  return (x_train, y_train), (x_test, y_test)

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