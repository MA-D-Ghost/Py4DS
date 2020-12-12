'''
Ho ten: Ton Thien Minh Anh
MSSV: 18110049
Bai thuc hanh Lab07 - TensorFlow: Basic classification: Classify images of clothing

Tutorial Link: https://www.tensorflow.org/tutorials/keras/classification
'''

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# =============== IMPORT FASHION DATASET ===============

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# =============== EXPLORE THE DATA ===============

print('\n ===== EXPLORE THE DATA ===== \n')

print('Train Images Shape: ',train_images.shape)
print('Train Labels Len: ',len(train_labels))
print('Train Labels: ',train_labels)
print('Test Images Shape: ',test_images.shape)
print('Test Labels Len: ',len(test_labels))

# =============== PREPROCESS THE DATA ===============

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale these values to a range of 0 to 1 before
#  feeding them to the neural network model. 
train_images = train_images / 255.0

test_images = test_images / 255.0

# Display the first 25 images from the training set 
# and display the class name below each image

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# =============== BUILD THE MODEL ===============

# SET UP THE LAYERS

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# COMPILE THE MODEL

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# =============== TRAIN THE MODEL ===============

# Training the neural network model requires the following steps:

# 1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
# 2. The model learns to associate images and labels.
# 3. You ask the model to make predictions about a test set—in this example, the test_images array.
# 4. Verify that the predictions match the labels from the test_labels array.

# FEED THE MODEL

print('===== Accuracy of train Model =====')
model.fit(train_images, train_labels, epochs=10)

# EVALUATE ACCURACY
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# MAKE PREDICTIONS

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)


# A prediction is an array of 10 numbers. 
# They represent the model's "confidence" that the image corresponds
#  to each of the 10 different articles of clothing. 
# You can see which label has the highest confidence value:
print('===== Make Predictions =====')
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])


# Graph to look at the full set of 10 class predictions.
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# VERIFY PREDICTIONS

# With the model trained, we can use it to make predictions about some images.

# The 0th image
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# The 12th image
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
