from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
import json
import numpy as np
from scipy import ndimage
from scipy import misc
from sklearn import preprocessing
from sklearn.utils import shuffle
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import csv
import cv2

image_width = 320
#image_height = 160
image_height = 75  # masked image dimension
image_depth = 3
batch_size = 128
nb_batches = 1
nb_validation_batches = 1
step_ratio = 20  # steering angle multipler
nb_classes = 1 + (2 * step_ratio)  # -step_ratio, 0, +step_ratio
incremental = False
short_run = False
single_overlay = False
classification_only = False
add_shifted = False
add_rotated = False
add_side_cameras = False
validation_split = 0.2

plus5 = cv2.getRotationMatrix2D((80, 160), 5, 1)  # +5 degree rotation matrix
minus5 = cv2.getRotationMatrix2D((80, 160), -5, 1)  # -5 degree rotation matrix

# add training images for center camera with optional shift/rotate/side camera
def add_training(log_list_, file_name, skip_row0):
    with open(file_name, newline='') as csvfile:
        driving_log = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        c = 0
        for row in driving_log:
            if skip_row0 == False or c > 0:
                log_list_.append((row[0], row[3], 0))   # center unmodified
                if add_shifted:
                    log_list_.append((row[0], row[3], 1))  # left shift
                    log_list_.append((row[0], row[3], 2))  # right shift
                if add_rotated:
                    log_list_.append((row[0], row[3], 3))  # center -5 degree rotation
                    log_list_.append((row[0], row[3], 4))   # center +5 degree rotation
                if add_side_cameras:
                    log_list_.append((row[1], row[3], 0))   # left unmodified
                    log_list_.append((row[2], row[3], 0))   # right unmodified
            c += 1



def get_image(image, select):
    if select == 1:
        M = np.float32([[1,0,-3],[0,1,0]])
        image = cv2.warpAffine(image, M, (320, 160))
    elif select == 2:
        M = np.float32([[1,0,3],[0,1,0]])
        image = cv2.warpAffine(image, M, (320, 160))
    elif select == 3:
        image = cv2.warpAffine(image, minus5, (320, 160))  # size parameters reversed
    elif select == 4:
        image = cv2.warpAffine(image, plus5, (320, 160))  # size parameters reversed
    # else return image as-is
    return image



def generate_training():
    X = np.zeros((batch_size, image_height, image_width, image_depth))
    y = np.zeros((batch_size, 1), dtype=int)
    while 1:
        c = 0
        for r in range(batch_size * nb_batches):
            row = train_list[r]
            image = misc.imread(row[0])
            get_image(image, row[2])
            X[c] = image[60:135, :, :]
            y[c] = int(step_ratio + (step_ratio * float(row[1])))
            c += 1
            if c == batch_size:
                c = 0
                yield (X, y)
       


def generate_validation():
    X = np.zeros((batch_size, image_height, image_width, image_depth))
    y = np.zeros((batch_size, 1), dtype=int)
    while 1:
        c = 0
        for r in range(batch_size * nb_validation_batches):
            row = validate_list[r]
            image = misc.imread(row[0])
            get_image(image, row[2])
            X[c] = image[60:135, :, :]
            y[c] = int(step_ratio + (step_ratio * float(row[1])))
            c += 1
            if c == batch_size:
                c = 0
                yield (X, y)


 
# split layers for refinement learning (https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py)
preprocessing_layers = [
    BatchNormalization(input_shape=(image_height, image_width, image_depth))
]
# derived from nvidia dave-2 architecture (linked paper)
# added max pooling and dropout to minimize overfitting
feature_layers = [
    Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid'),
    MaxPooling2D(pool_size=(2,2), strides=(1,1)),
    ELU(),
    Dropout(0.25),
    Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid'),
    ELU(),
    Dropout(0.25),
    Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid'),
    ELU(),
    Convolution2D(64, 3, 3, border_mode='valid'),
    ELU(),
    Convolution2D(64, 3, 3, border_mode='valid'),
    ELU(),
    Flatten()
]
classification_layers = [
    Dense(1000),
    ELU(),
    Dropout(0.5),
    Dense(500),
    ELU(),
    #Dense(nb_classes),
    #Activation('softmax')
    Dense(1)
]


model = Sequential(preprocessing_layers + feature_layers + classification_layers)

# for refining the model, disable training on the feature detection layers
if incremental and classification_only:
    for l in feature_layers:
        l.trainable = False

log_list = []
if single_overlay:
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round5/e_bridge/driving_log.csv', False)
else:
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/1/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/2/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/3/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e1/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e2/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e3/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e4/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e5/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e6/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e7/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e8/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e9/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round5/1/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round5/e_dirt1/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round5/e_bridge/driving_log.csv', False)
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round5/e_rcurve/driving_log.csv', False)

print('list rows = {}'.format(len(log_list)))
training_size = len(log_list)
log_list = shuffle(log_list, random_state=0)
marker = int(training_size * (1.0 - validation_split))
train_list = log_list[0:marker]
validate_list = log_list[marker:]
nb_batches = len(train_list) // batch_size
nb_validation_batches = len(validate_list) // batch_size

# use keras normalization; excessive memory consumed using the code below prior to pipeline

# sparse_categorical_crossentropy seems to be the closes to tensorflow crossentropy with logits
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

#print(feature_layers[0].input_shape)
#print(feature_layers[1].output_shape)
#print(feature_layers[2].output_shape)
#print(feature_layers[3].output_shape)
#print(feature_layers[4].output_shape)
#print(feature_layers[5].output_shape)
#print(feature_layers[6].output_shape)
#print(feature_layers[7].output_shape)
#print(feature_layers[8].output_shape)
#print(feature_layers[9].output_shape)
#print(feature_layers[10].output_shape)
#print(feature_layers[11].output_shape)
#print(classification_layers[0].input_shape)

# if the incremental flag is set, load existing weights and run a short training on the classification layers only
if incremental:
    print('Loading weights')
    model.load_weights('model.h5', False)

if short_run:
    nb_epoch = 1
else:
    nb_epoch = 5
history = model.fit_generator(generate_training(), samples_per_epoch=(batch_size * nb_batches), nb_epoch=nb_epoch,
                              validation_data=generate_validation(), nb_val_samples=(batch_size * nb_validation_batches))

# save the model design
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()

# save the trained network weights
model.save('model.h5')

