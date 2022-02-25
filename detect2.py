import cv2
import os
import numpy as np
import dlib
from datetime import datetime
import time
import requests
from keras_vggface.vggface import VGGFace
from PIL import Image
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
class detector:
    EMBEDDINGS_KEY = 'FaceNet/embeddings.npy'
    LABELS_KEY = 'FaceNet/labels.npy'
    FNDetector = MTCNN()
    FNModel = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    labels, embeddings = np.load(LABELS_KEY), np.load(EMBEDDINGS_KEY)
    """Class that runs detections and objects tracking for a frame"""
    trackers = []
    writer   = None
    camName  = None
    fourcc   = None
    objColor =  (150, 150, 100)
    frameNum    = 0
    fps         = 6
    timestamp   = 0
    timesMissed = 0
    mysql       = None

    def __init__(self, camName, protocol, model):
        self.fourcc     = cv2.VideoWriter_fourcc(*'avc1')
        self.net        = cv2.dnn.readNet("F:\DoAn\Home\SecurityCameraAI\Model\yolov4-tiny.weights", "F:\DoAn\Home\SecurityCameraAI\Model\yolov4-tiny.cfg")
        self.camName    = camName

    def updateTrackers(self, frame):
        for trackerData in self.trackers:
            frame = self.updateTracker(frame, trackerData)
        return frame

    def updateTracker(self, frame, trackerData):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker = trackerData['tracker']
        color    = trackerData['color']
        label    = trackerData['label']
        tracker.update(rgb)
        pos     = tracker.get_position()
        startX  = int(pos.left())
        startY  = int(pos.top())
        endX    = int(pos.right())
        endY    = int(pos.bottom())
        self.updateFrame(frame, [startX, startY, endX, endY], label, color)
        return frame

    def startTracker(self, frame, box, label, color):
        tracker = dlib.correlation_tracker()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        tracker.start_track(rgb, rect)
        self.trackers.append({'tracker': tracker, 'color': color, 'label': label})

    def detect(self, frame, fps):

        if frame is None:
            return
        #Count frames for FPS calculation
        self.frameNum = self.frameNum + 1
        if self.timestamp != 0:
            self.fps = 1/(time.time() - self.timestamp)
        self.timestamp = time.time()

        #If there were no dlib correlation trackers or we just got 20 frames passed
        if not self.trackers  or self.frameNum == 5:
            # if self.frameNum < 5:
            #     return frame
            self.frameNum = 0

            detections = self.detectObjectsYolo(frame)
            if not detections:
                #In case object detection missed the object, give it another chance (5 chances, actually)
                if self.trackers is not None and self.timesMissed < 15:
                    self.timesMissed = self.timesMissed + 1
                    frame = self.updateTrackers(frame)
                    return frame

                self.timesMissed = 0
                self.trackers = []
                if self.writer is not None:
                    self.writer.release()
                    self.writer = None
                return frame

            self.trackers = []

            for detection in detections:
                label = ""
                box   = detection['box']
                color = self.objColor
                self.startTracker(frame, box, label, color)
                frame = self.updateFrame(frame, box, label, color)
            self.writeFrame(frame, fps, flag)
            return frame
        frame = self.updateTrackers(frame)
        self.writeFrame(frame, fps, False)
        return frame

    def updateFrame(self, frame, box, label, color):
        """Draw a box around the detected object
        """
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        textSize, baseline = cv2.getTextSize( label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        # cv2.rectangle(frame, (box[0], box[1]) , (box[0] + textSize[0], box[1] - textSize[1] - 17), color, -1)
        # cv2.putText(frame, label, (box[0], box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame
    

    def writeFrame(self, frame, fps, flag):
        nowDate = datetime.now().strftime("%Y-%m-%d")
        nowTime = datetime.now().strftime("%H-%M-%S")
        if self.writer is None:
            if not os.path.exists('F:/DoAn/data_video/' + nowDate):
                os.makedirs('F:/DoAn/data_video/' + nowDate)
            (h, w) = frame.shape[:2]
            self.writer = cv2.VideoWriter('F:/DoAn/data_video/' + nowDate + '/ai-' + nowTime + '.mp4',self.fourcc, fps, (w, h), True)
            try:
                print('http://localhost:8000/api/video?name=ai-{}.mp4&flag={}'.format(nowTime, str(flag)))
                requests.get('http://localhost:8000/api/video?name=ai-{}.mp4&flag={}'.format(nowTime, str(flag)))
            except:
                print("Cannot connect to server")
        self.writer.write(frame)

    def detectObjects(self, frame):
        """Run objects detection on the frame. In case of detecting objects return list of arrays
           (boxes coordinates), object id and object type
        """
        (h, w) = frame.shape[:2]
        fx = 300
        fy = 300
        mean   = cv2.mean(frame)

        #Make input image square to avoid geometric distortions
        frame  = cv2.copyMakeBorder(frame, 0, w - h, 0, 0, cv2.BORDER_CONSTANT, round(max(mean)))
        blob   = cv2.dnn.blobFromImage(frame,  1/max(mean), (fx, fy), mean, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        detected   = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.95 and int(detections[0, 0, i, 1]) in self.classesOfInterest:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, w, w, w])
                detected.append({'box':box.astype("int"), 'confidence': confidence, 'idx': idx})
        return detected

    def detectObjectsYolo(self, frame):
        classes = ['Thanh']
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        minscore = 0.28
        minindex = -1
        (h, w) = frame.shape[:2]
        mean   = cv2.mean(frame)

        #Make input image square to avoid geometric distortions
        frame  = cv2.copyMakeBorder(frame, 0, w - h, 0, 0, cv2.BORDER_CONSTANT, round(max(mean)))
        blob   = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        
        outs = self.net.forward(output_layers)
        detected   = []
        confidences   = []
        boxes   = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classID  = np.argmax(scores)
                print(classID)
                confidence = scores[classID ]
                if confidence > 0.9:
                # Object detected
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Rectangle coordinates
                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    # box = [x, y, x + int(width), y + int(height)]

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            x, y, width, height = boxes[i]
            box = [x, y, int(x) + int(width), int(y) + int(height)]
            # extract the face
            face = frame[y:int(y) + int(height), x:int(x) + int(width)]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize((224, 224))
            face_array = np.asarray(image)
            samples = np.asarray([face_array], 'float32')
            samples = preprocess_input(samples, version=2)
            yhat = self.FNModel.predict(samples)
            if(len(yhat) > 0):
                for index in range(len(self.embeddings)):
                    score = cosine(self.embeddings[index], yhat[0])
                    if score < minscore:
                        minscore = score
                        minindex = index
                minindex = -1 if minscore > 0.28 else minscore 

            detected.append({'box':  box, 'confidence': float(confidences[i]), 'idx': 3})
        flag = False if minindex == -1 else True
        return detected, flag



        # return detected, boxes, confidences