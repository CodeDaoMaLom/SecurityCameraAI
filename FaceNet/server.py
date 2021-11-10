from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from flask import Flask, request, jsonify, abort
from flask.wrappers import Response
from flask_cors import CORS, cross_origin
from tensorflow.python.keras.backend import set_session
import matplotlib.image as mpimg
import tensorflow as tf
import recognize
import numpy as np
import base64
import re
import io
import cv2
import os
import shutil
import gen_image
import pandas as pd
import json

app = Flask(__name__)
CORS(app, support_credentials=True)

EMBEDDINGS_KEY = 'FaceNet/embeddings.npy'
LABELS_KEY = 'FaceNet/labels.npy'
EMBEDDINGS_KEY_BAK = 'FaceNet/embeddings.npy.bak'
LABELS_KEY_BAK = 'FaceNet/labels.npy.ank'

labels, embeddings = recognize.load_emb(LABELS_KEY, EMBEDDINGS_KEY)

global detector
global model

# create the detector, using default weights
graph = tf.get_default_graph()
sess = tf.Session()
set_session(sess)

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

print("Length labels and embeddedings")
print(len(labels))
print(len(embeddings))

@app.route('/verify-face', methods=['POST'])
def verify_face():
    global sess
    global graph
    b64img = request.json
    base64_data = re.sub('^data:image/.+;base64,', '', b64img["data"])
    decoded_data = base64.b64decode(base64_data)
    byteimage = io.BytesIO(decoded_data)
    byteimage = pyplot.imread(byteimage, format='JPG')
    byteimage = cv2.cvtColor(byteimage, cv2.COLOR_RGB2BGR)


    with graph.as_default():
        set_session(sess)
        matchidx, score = recognize.is_match(detector, model, embeddings, byteimage)
        if matchidx is -1:
           return jsonify({ "person": "", "probability": "0"})
        label = labels[matchidx]
        print('='*50)
        print(matchidx, label, score)
        index = label.find('_')
        personId = label[0:index]

        return jsonify({ "person": personId, "probability": "%.2f"%(1 - score)})         

@app.route('/setup-face', methods=['POST'])
def setup_face():
    global labels
    global embeddings
    global sess
    global graph

    b64img = request.json
    base64_data = re.sub('^data:image/.+;base64,', '', b64img["data"])
    decoded_data = base64.b64decode(base64_data)
    byteimage = io.BytesIO(decoded_data)
    byteimage = pyplot.imread(byteimage, format='JPG')
    byteimage = cv2.cvtColor(byteimage, cv2.COLOR_RGB2BGR)

    try:
        for i in range(8):
            with graph.as_default():
                set_session(sess)
                byteimage = gen_image.transforms[i](byteimage)
                emb, label = recognize.setup_emb(detector, model, b64img["name"], byteimage)
                labels = np.append(labels, label)
                embeddings = np.vstack((embeddings, emb))

    except Exception as e:
        print('Error: \n', str(e))
        abort(500, str(e))

    try:
        if (len(embeddings) != len(labels)):
            raise MissMatchException("Length of Setup data miss match")

        shutil.copy2(EMBEDDINGS_KEY, EMBEDDINGS_KEY_BAK)
        shutil.copy2(LABELS_KEY, LABELS_KEY_BAK)
        np.save(EMBEDDINGS_KEY, embeddings)
        np.save(LABELS_KEY, labels)
        after_labels, after_embeddings = recognize.load_emb(LABELS_KEY, EMBEDDINGS_KEY)

        if (len(after_embeddings) != len(after_labels)):
            os.remove(EMBEDDINGS_KEY)
            os.remove(LABELS_KEY)
            os.rename(EMBEDDINGS_KEY_BAK, EMBEDDINGS_KEY)
            os.rename(LABELS_KEY_BAK, LABELS_KEY)
            raise MissMatchException("Length lables and embeddings after setup miss match")
        else:
            os.remove(EMBEDDINGS_KEY_BAK)
            os.remove(LABELS_KEY_BAK)

    except Exception as e:
        print('Error: \n', str(e))
        abort(500, str(e))

    return 'OK'

@app.route('/get-trained-faces', methods=['GET'])
def getTrainedFaces():
    listData = []
    for i in range(0, len(labels)):
        dictData = {}
        dictData['name'] = labels[i].split('_')[0]
        # dictData['data'] = {"no": i, "timestamp": labels[i].split('_')[1]}
        dictData['data'] = i
        listData.append(dictData)

    df = pd.DataFrame(listData, columns=['name', 'data'])
    df = df.groupby(['name'])['data'].apply(list).reset_index(name="data")
    out = df.to_json(orient="records", force_ascii=False)
    return Response(out, mimetype='application/json')

@app.route('/delete-trained-faces', methods=['GET'])
def deleteTrainedFaces():
    global labels
    global embeddings

    embeddingsList = embeddings.tolist()
    needDelete = json.loads(request.args.get('ids'))

    try:
        for i in range(0, len(needDelete)):
            delete  = needDelete[i] - i if i != 0 else needDelete[i]
            print(labels[delete])
            labels = np.delete(labels, delete)
            embeddingsList.pop(delete)
    except Exception as e:
        print('Error: ', str(e))
        return Response(str(e), status=500)
    return Response("Done", status=200)

class MissMatchException(Exception):
    def __init__(self, value):
        self.value = value

if __name__ == '__main__':
   app.run(host="0.0.0.0", port="4444", threaded=True, debug=True)