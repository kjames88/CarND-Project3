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
step_ratio = 10  # steering angle multipler
nb_classes = 1 + (2 * step_ratio)  # -step_ratio, 0, +step_ratio
incremental = True
classification_only = False
add_shifted = True
add_rotated = True
add_side_cameras = False

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
    Dense(100),
    ELU(),
    Dropout(0.5),
    Dense(50),
    ELU(),
    Dense(nb_classes),
    Activation('softmax')
]

model = Sequential(preprocessing_layers + feature_layers + classification_layers)

# for refining the model, disable training on the feature detection layers
if incremental and classification_only:
    for l in feature_layers:
        l.trainable = False

log_list = []
if incremental:  # just load one set of training data to refine existing network weights
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round4/e8/driving_log.csv', False)
else:  # observed results markedly better with a single pass vs refining, which seems to overweight the new data 
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
    add_training(log_list, '/home/kjames/code/udacity/nd013/project3/round5/1/driving_log.csv', False)

print('list rows = {}'.format(len(log_list)))
training_size = len(log_list)
X_train = np.zeros((training_size, image_height, image_width, image_depth), dtype=np.uint8)
y_train = np.zeros(training_size, dtype=int)
print('X_train shape {}'.format(X_train.shape))
#my_hist = np.zeros(nb_classes, dtype=int)
plus5 = cv2.getRotationMatrix2D((80, 160), 5, 1)  # +5 degree rotation matrix
minus5 = cv2.getRotationMatrix2D((80, 160), -5, 1)  # -5 degree rotation matrix
for r in range(training_size):
    image = misc.imread(log_list[r][0])
    if log_list[r][2] == 1:
        M = np.float32([[1,0,-3],[0,1,0]])
        image = cv2.warpAffine(image, M, (320, 160))
    elif log_list[r][2] == 2:
        M = np.float32([[1,0,3],[0,1,0]])
        image = cv2.warpAffine(image, M, (320, 160))
    elif log_list[r][2] == 4:
        image = cv2.warpAffine(image, plus5, (320, 160))  # size parameters reversed
    elif log_list[r][2] == 3:
        image = cv2.warpAffine(image, minus5, (320, 160))  # size parameters reversed
    X_train[r] = image[60:135, :, :]
    y_train[r] = int(step_ratio + (step_ratio * float(log_list[r][1])))
    #my_hist[y_train[r]] += 1
#print('my_hist {}'.format(my_hist))
print('Read completed')
classes = np.zeros(nb_classes, dtype=int)
for i in range(nb_classes):
    classes[i] = i
#y_train_one_hot = preprocessing.label_binarize(y_train, classes)

# shuffle the data
X_train, y_train = shuffle(X_train, y_train, random_state=0)
print('Shuffle completed')

# use keras normalization; excessive memory consumed using the code below prior to pipeline
# normalization to color range
#a = -0.5
#b = 0.5
#color_range = 255
#X_normalized = a + ((X_train * (b - a)) / color_range)
#print('Normalization completed')

#test = misc.imread('/home/kjames/code/udacity/nd013/project3/data/' + log_list[0][0])
#plt.imshow(test)

# sparse_categorical_crossentropy seems to be the closes to tensorflow crossentropy with logits
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# if the incremental flag is set, load existing weights and run a short training on the classification layers only
if incremental:
    print('Loading weights')
    model.load_weights('model.h5', False)

if incremental:
    nb_epoch = 1
else:
    nb_epoch = 25
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2)

# save the model design
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()

# save the trained network weights
model.save('model.h5')

