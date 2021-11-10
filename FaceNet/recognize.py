import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot
from PIL import Image
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
from datetime import datetime
import numpy as np
import cv2
import os

THRESHOLD = 0.28

# extract a single face from a given photograph
def extract_face(detector, pixels, mode = False, name = '', required_size=(224, 224)):
	# detect faces in the image
    results = detector.detect_faces(pixels)
    if(len(results) <= 0):
        raise ValueError("Can not detect face")
	# extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    if mode:
        try:
            cv2.imwrite('faces/' + name + '.jpg', face_array)
            return face_array
        except Exception as e:
            print('Error: ', e)
    else:
	    return face_array

def get_embedding_img(detector, model, img, mode = False, name = ''):
    try:
        face = extract_face(detector, img, mode, name)
    except Exception as e:
        print('Error: ', e)
        yhat = []
        return yhat
    samples = np.asarray([face], 'float32')
    samples = preprocess_input(samples, version=2)
    yhat = model.predict(samples)

    return yhat
 
# determine if a candidate face is a match for a known face
def is_match(detector, model, known_embeddings, candidate_img, thresh=THRESHOLD):
    # calculate distance between embeddings
    minscore = thresh
    minindex = -1
    candidate_embedding = get_embedding_img(detector, model, candidate_img)
    if(len(candidate_embedding) > 0):
        for index in range(len(known_embeddings)):
            score = cosine(known_embeddings[index], candidate_embedding[0])
            if score < minscore:
                minscore = score
                minindex = index

    #return score
    if minscore < thresh:
        print('face is MATCH (%.3f <= %.3f)' % (minscore, thresh))
    else:
        print('face is NOT MATCH (%.3f > %.3f)' % (minscore, thresh))
        minindex = -1
        
    return minindex, minscore
 
def setup_emb(detector, model, personname, filenames):
    sectime = int(datetime.now().timestamp())
    label_name = '%s_%s'%(personname, sectime)
    emb = get_embedding_img(detector, model, filenames, False, label_name)

    return emb, label_name    

def load_emb(lblname, embname):
    return np.load(lblname), np.load(embname)

