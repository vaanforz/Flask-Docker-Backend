from flask import Flask,render_template
import socket

import os, sys
import numpy as np
from PIL import Image 

from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from keras.applications.xception import Xception

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

from models.darknet import darknet 

#sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

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
    y = model.predict(x)
    return y

def analyze_pic(pic_path):

    r = darknet.detect(det_net, det_meta, pic_path)
    #img = Image.open(pic_path)
    img = Image.open(io.BytesIO(pic_path)).convert("RGB")
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

        # y_min,x_min,y_max,x_max
        jbox['detection_box'] = [max(0,upper/height),max(0,left/width),
                                 min(1,down/height),min(1,right/width)]

        result['predictions'].append(jbox)

    return result

@app.route("/")
def index():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return render_template('index.html', hostname=host_name, ip=host_ip)
    except:
        return render_template('error.html')

@app.route("/predict", methods=['POST'])
def predict():
	image_data = request.files['image']
	return analyze_pic(image_data.read())


if __name__ == "__main__":
    det_net = darknet.load_net("/app/app/models/darknet/cfg/yolov3-food.cfg", "/app/app/models/darknet/backup/food/yolov3-food_final.weights", 0)
    det_meta = darknet.load_meta("/app/app/models/darknet/cfg/food.data")

    classes = 231
    base_model = Xception(include_top=True, input_shape=(299, 299, 3))
    base_model.layers.pop()
    predictions = Dense(classes, activation='softmax')(base_model.layers[-1].output)
    clf_model = Model(input=base_model.input, output=[predictions])
    clf_model.load_weights("/app/app/models/classification/models/xception-0-15-0.82.h5")
    print("[*]Loaded object detection and classification model!")
    class_dict = {v:k for k,v in np.load("/app/app/models/classification/class_index/food231.npy")[()].items()}

    app.run(host='0.0.0.0', port=5000)
