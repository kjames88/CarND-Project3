import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :]
    X_test = np.zeros((1, 75, 320, 3))
    X_test = transformed_image_array[:, 60:135, :, :]
    print('image shape {} {}'.format(transformed_image_array.shape, X_test.shape))
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    # Normalization
    #a = -0.5
    #b = 0.5
    #transformed_image_array = a + ((transformed_image_array * (b - a)) / 255)

    step_ratio = 10
    steering_angle_pred = model.predict(X_test, batch_size=1)
    #print('pred type is {}'.format(type(steering_angle_pred)))
    steering_angle_pred_int = step_ratio
    peak = np.max(steering_angle_pred)
    for i in range(steering_angle_pred.shape[1]):
        if steering_angle_pred[0][i] == peak:
            steering_angle_pred_int = i
    #print('peak is {} pred is {} {}'.format(peak, steering_angle_pred, steering_angle_pred_int))
    steering_angle = -1.0 + (float(steering_angle_pred_int) / float(step_ratio))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = max(0.1, -0.15/0.05 * abs(steering_angle) + 0.35)
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
