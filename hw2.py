# %% [markdown]
# # **HOMEWORK 2 batch version**

# %%
#%history -f output.txt

# %% [markdown]
# # **Libraries importation**

# %%
import os
import cv2
import scipy
import numpy as np 
import tensorflow as tf
from PIL import Image
from collections import defaultdict
from matplotlib import pyplot as plt 
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('libraries imported')

# %% [markdown]
# # **settings**

# %%
# Avoid OOM errors by setting GPU memory consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# %% [markdown]
# # **Data Collection**

# %% [markdown]
# ## **Data Augmentation**

# %%
datagen_train = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2
)

datagen_val = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# %% [markdown]
# ## **Data Load**

# %% [markdown]
# ### **Load without validation**

# %%
Train = datagen.flow_from_directory(
    'train',
    target_size=(256,256),
    batch_size=32,
    class_mode='categorical'
)

train_labels = []
train_labels = Train.classes
num_classes = Train.num_classes
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)

# %% [markdown]
# ### **Load with validation**

# %%
Train = datagen_train.flow_from_directory(
    'train',
    target_size=(256,256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

Val = datagen_val.flow_from_directory(
    'train',
    target_size=(256,256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
train_labels = []
train_labels = Train.classes
num_classes = Train.num_classes
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)


# %% [markdown]
# ## **Visualization of the dataset**

# %%
def visualize_dataset(dataset, num_samples=5):
    for images, labels in dataset:
        num_samples_batch = min(num_samples, len(images))
        fig, ax = plt.subplots(1, num_samples_batch, figsize=(20, 20))
        
        for i in range(num_samples_batch):
            ax[i].imshow((images[i] * 255).astype("uint8"))  # Remove the rescaling here
            ax[i].set_title(f"Label: {labels[i]}")
            ax[i].axis("off")
        
        plt.show()
        break


# num_samples = 5
# for images, labels in Train.take(1):
#     num_samples_batch = min(num_samples, len(images))
#     fig, ax = plt.subplots(1, num_samples_batch, figsize=(20, 20))
#     for i in range(num_samples):
#         ax[i].imshow(images[i].numpy().astype("uint8"))
#         ax[i].set_title(f"Label: {labels[i]}")
#         ax[i].axis("off")
#     plt.show()

# %%
visualize_dataset(Train)

# %% [markdown]
# # **Data Preprocessing**
# -    Resize images to a common size (e.g., 96x96, as mentioned in the description).
# -    Normalize pixel values to a range between 0 and 1.
# -    Consider data augmentation techniques (e.g., rotation, flipping) to increase the diversity of your training set.    

# %% [markdown]
# # **Model Selection:**
# ## Model Design
# -    Define your own CNN architecture. Start with a simple architecture and gradually increase complexity if needed.
# -    Experiment with different layer configurations, activation functions, and filter sizes.
# -    Consider incorporating dropout for regularization.

# %% [markdown]
# ## **Approach 1**
# -   Define the first approach with a specific architecture, optimizer, and regularization techniques.
# -   Choose appropriate hyperparameters (learning rate, batch size, etc.).
# -   Train the model on the training set and evaluate on the test set.
# -   Collect and analyze metrics such as accuracy, precision, recall, and F1 score.

# %%
model = Sequential()

# %%
# Layers
# (3,3) is the pixel selection, 1 is the translation of pixels
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5)) # Dropout layer to reduce overfitting

num_classes = 5
model.add(Dense(num_classes, activation='softmax'))

# %%
optimizer = Adam(learning_rate=0.001)

# %% [markdown]
# **Validation ON**

# %%
metrics = ['accuracy', 'val_accuracy']

# %% [markdown]
# **Validation OFF**

# %%
metrics = ['accuracy']

# %%
model.compile(optimizer, loss=tf._losses.CategoricalCrossentropy(), metrics=metrics)

# %%
model.summary()

# %% [markdown]
# ## **Approach 2**
# -    Define the second approach with a different architecture, optimizer, or regularization techniques.
# -    Adjust hyperparameters independently of the first approach.
# -    Train the model on the training set and evaluate on the test set.
# -    Collect and analyze metrics as done for the first approach.

# %%
model = Sequential()

# %%
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3), kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

num_classes = 5
model.add(Dense(num_classes, activation='softmax'))

# %%
beta_1 = 0.9
beta_2 = 0.999
optimizer = Adam(learning_rate=0.001, beta_1=beta_1, beta_2=beta_2)

# %% [markdown]
# **Validation ON**

# %%
model.compile(optimizer, loss=tf._losses.CategoricalCrossentropy(), metrics=['accuracy','val_accuracy'])

# %% [markdown]
# **Validation OFF**

# %%
model.compile(optimizer, loss=tf._losses.CategoricalCrossentropy(), metrics=['accuracy'])

# %%
model.summary()
for layer in model.layers:
    print(layer.output_shape)

# %% [markdown]
# ## **Hyperparameter Analysis**
# -   Choose at least one hyperparameter (e.g., learning rate) and perform a systematic analysis.
# -   Train models with different values of the chosen hyperparameter.
# -   Compare and visualize the impact on metrics.
# -   Consider to apply an early stopping of the training in order to avoid overfitting (see slide 11 pag 55)
# -   Consider if to apply Dropout or parameter sharing
# 

# %% [markdown]
# # **Model Training**

# %%
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# %%
class_labels = np.unique(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(zip(class_labels, class_weights))

# %% [markdown]
# ### **Fitting with early stopping**

# %% [markdown]
# #### **validation OFF**

# %%
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
hist = model.fit(Train, epochs=20, callbacks=[tensorboard_callback, early_stopping], class_weight=class_weights_dict)

# %% [markdown]
# #### **validation ON**

# %%
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
hist = model.fit(Train, epochs=20, callbacks=[tensorboard_callback, early_stopping], validation_data=Val, class_weight=class_weights_dict)

# %% [markdown]
# ### **Fitting without early stopping**

# %% [markdown]
# #### **Validation OFF**

# %%
hist = model.fit(Train, epochs=20, callbacks=[tensorboard_callback], class_weight=class_weights_dict)

# %% [markdown]
# #### **Validation ON**

# %%
hist = model.fit(Train, epochs=20, callbacks=[tensorboard_callback], validation_data=Val, class_weight=class_weights_dict)

# %% [markdown]
# ## **Plotting Model Performance**

# %% [markdown]
# ### **Validation ON**

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(hist.history['loss'], color='teal', label='train_loss')
ax1.plot(hist.history['val_loss'], color='orange', label='val_loss')  # Validation loss
ax1.set_title('Loss')
ax1.legend()

ax2.plot(hist.history['accuracy'], color='teal', label='train_accuracy')
ax2.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')  # Validation accuracy
ax2.set_title('Accuracy')
ax2.legend()

plt.show()

# %% [markdown]
# ### **Validation OFF**

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(hist.history['loss'], color='teal', label='train_loss')
ax1.set_title('Loss')
ax1.legend()

ax2.plot(hist.history['accuracy'], color='teal', label='train_accuracy')
ax2.set_title('Accuracy')
ax2.legend()

plt.show()

# %% [markdown]
# # **Test**

# %%
Test = tf.keras.utils.image_dataset_from_directory('test')
test_iterator = Test.as_numpy_iterator()

all_X_test = []
all_y_test = []

for test_batch in test_iterator:
    X_test_batch, y_test_batch = test_batch
    X_test_normalized_batch = X_test_batch/255.0
    yhat_batch = model.predict(X_test_normalized_batch)
    all_X_test.append(X_test_normalized_batch)
    all_y_test.append(y_test_batch)

X_test_normalized = np.concatenate(all_X_test)
y_test = np.concatenate(all_y_test)

test_iterator = Test.as_numpy_iterator()
Test_batch = next(test_iterator)
X_test, y_test = Test_batch
X_test_normalized = X_test/255.0

# %%
yhat = model.predict(X_test_normalized)

# %%
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# %%
for batch in Test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    pre.update_state(y_one_hot, yhat)
    re.update_state(y_one_hot, yhat)
    acc.update_state(y_one_hot, yhat)                                    

# %%
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

# %% [markdown]
# # **Results Visualization and Comparison**
# -    Create visualizations (tables, charts, graphs) to present your results.
# -    Provide detailed commentary on each visualization, explaining trends or differences observed.

# %% [markdown]
# ## **Comparison on accuracy between methods**

# %% [markdown]
# ### **On train and validation**

# %%
fig = plt.figure(figsize=(16, 8))
plt.plot(hist.history['accuracy'], color='teal', label='train_accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
plt.suptitle('Model accuracy comparison on train and validation', fontsize=14)
plt.legend(loc="upper left")
plt.show()

# %% [markdown]
# ### **On Test**

# %%
fig=plt.figure(figsize=(16, 8))
fig = plt.figure(figsize=(16, 8))
plt.plot(acc.result, color='teal', label='test_accuracy') # to edit
plt.suptitle('Model accuracy comparison on test', fontsize=14)
plt.legend(loc="upper left")
plt.show()

# %% [markdown]
# ## **Fine-Tuning**

# %% [markdown]
# ## **Deployment**

# %%


# %% [markdown]
# ##valutazioni da fare poi:
# - regularization per ridurre l'overfitting?
# - il numero di images cambia da classe  a classe (train) vedere se serve prenderne un numero uguale per ciascuna classe
# - 


