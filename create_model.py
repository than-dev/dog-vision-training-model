import tensorflow as tf
import tensorflow_hub as hub
import tf_keras

IMG_SIZE = 224
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channel
MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

def create_model(breeds):
  print("Building model with:", MODEL_URL)

  model = tf_keras.Sequential([
      hub.KerasLayer(MODEL_URL), # Layer 1 (Input Layer)
      tf_keras.layers.Dense(units=len(breeds),
                            activation = "softmax" ) # Layer 2 (Output Layer)
  ])

  model.compile(
      loss = tf_keras.losses.SparseCategoricalCrossentropy(),
      optimizer = tf_keras.optimizers.Adam(),
      metrics = ["accuracy"]
  )

  model.build(INPUT_SHAPE)

  return model