
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

train_files, train_targets = load_dataset('/content/drive/MyDrive/train')
valid_files, valid_targets = load_dataset('/content/drive/MyDrive/valid')
test_files, test_targets = load_dataset('/content/drive/MyDrive/test')

len(train_files)

len(valid_files)

len(test_files)



print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

bottleneck_features = np.load('/content/drive/MyDrive/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

import tensorflow as tf

Resnet50_model = tf.keras.models.Sequential()
Resnet50_model.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(tf.keras.layers.Dense(1024, activation='relu'))
Resnet50_model.add(tf.keras.layers.Dense(133, activation='softmax'))

Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/weights_best_Resnet50.hdf5', 
                               verbose=1, save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=2, monitor='val_loss')
Resnet50_model.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=50, batch_size=20, callbacks=[checkpointer, early_stopping, reduce_lr], verbose=1)### TODO: Train the model.

### TODO: Calculate classification accuracy on the test dataset.
# get index of predicted dog breed for each image in test set
Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

import tensorflow as tf
import os
import numpy as np
from keras.preprocessing import image
import cv2

def label_to_category_dict(path):
    '''Returns a dictionary that maps labels to categories'''
    categories = os.listdir('/content/drive/MyDrive/train')
    label_to_cat = map(lambda x: (int(x.split('.')[0]) - 1, x.split('.')[1]), categories)
    label_to_cat = {label: category for label, category in label_to_cat}
    return label_to_cat

label_to_cat = label_to_category_dict(train_files)

def predict_breed(img_path):
    '''Predicts the breed of the given image'''
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    bottleneck_feature = tf.keras.models.Sequential([
                            tf.keras.layers.GlobalAveragePooling2D(input_shape=bottleneck_feature.shape[1:])
                        ]).predict(bottleneck_feature).reshape(1, 1, 1, 2048)
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return label_to_cat[np.argmax(predicted_vector)]

def extract_Resnet50(tensor):
    '''Returns the VGG16 features of the tensor'''
    return tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False).predict(tf.keras.applications.resnet50.preprocess_input(tensor))

predict_breed('/content/drive/MyDrive/Akita_00258.jpg')

predict_breed('/content/drive/MyDrive/Collie_03849.jpg')
