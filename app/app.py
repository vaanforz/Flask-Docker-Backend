from flask import Flask,render_template,request
import socket

import os, sys
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from keras.applications.xception import Xception

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

from models.darknet import darknet 

config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)

app = Flask(__name__)

def predict(model, img):
    width_height_tuple = (299, 299)
    if(img.size != width_height_tuple):
        img = img.resize(width_height_tuple, Image.NEAREST)
    x = img_to_array(img)
    x /= 255 * 1.
    x = x.reshape((1,) + x.shape)
    with graph.as_default():
        y = model.predict(x)
    return y

def analyze_pic(pic_path):
    img = (Image.open(pic_path)).convert('RGB')
    r = darknet.detect(det_net, det_meta, img_to_array(img))
    width = img.size[0]
    height = img.size[1]

    result = {}
    result['status'] = "ok"
    result['predictions'] = []

    for index,box in enumerate(r):
        prob = box[1]
        x,y,w,h = box[2][0],box[2][1],box[2][2],box[2][3]
        left = x-w/2
        upper = y-h/2
        right = x+w/2
        down = y+h/2
        cropped = img.crop((x-w/2,y-h/2,x+w/2,y+h/2))  # (left, upper, right, lower)
        y = predict(clf_model, cropped)

        class_id = np.argsort(y[0])[::-1][0]
        str_class = class_dict[class_id]

        jbox = {}
        jbox['label_id'] = str(class_id)
        jbox['label'] = str(str_class)
        jbox['probability'] = prob

        # y_min,x_min,y_max,x_max
        jbox['detection_box'] = [max(0,upper/height),max(0,left/width),
                                 min(1,down/height),min(1,right/width)]

        result['predictions'].append(jbox)

    return result

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/model/predict", methods=['POST'])
def detect_and_predict():
    img_data = request.files['image']
    return analyze_pic(img_data)


if __name__ == "__main__":
    global det_net, det_meta
    det_net = darknet.load_net(b"/app/app/models/darknet/cfg/yolov3-food.cfg", b"/app/app/models/darknet/backup/food/yolov3-food_final.weights", 0)
    det_meta = darknet.load_meta(b"/app/app/models/darknet/cfg/food.data")

    classes = 231
    base_model = Xception(include_top=True, input_shape=(299, 299, 3))
    base_model.layers.pop()
    predictions = Dense(classes, activation='softmax')(base_model.layers[-1].output)

    global clf_model, graph, class_dict
    clf_model = Model(input=base_model.input, output=[predictions])
    clf_model.load_weights("/app/app/models/classification/models/xception-0-15-0.82.h5")
    clf_model._make_predict_function()
    graph = tf.get_default_graph()

    class_dict = {v:k for k,v in np.load("/app/app/models/classification/class_index/food231.npy")[()].items()}

    app.run(host='0.0.0.0', port=5000)
