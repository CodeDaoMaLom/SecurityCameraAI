import webcam
from pprint import pprint
import detect
import detect2
from multiprocessing import Process, Manager
import imagezmq
import numpy as np
import cv2
import time
import threading
import configparser
from flask import Flask, Response
from werkzeug.serving import run_simple

app = Flask(__name__)
app.debug = True

config = configparser.ConfigParser()
config.read('config.ini')

camName = config.get('cameras', 'cameras')
cam     = config.get(camName, 'url')

imagezmqPort = 5555
ports = []
output = {}

def prepMultipart(frame):
    jpg = cv2.imencode('.jpg', frame)[1]
    return b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tostring()+b'\r\n'

def runDetectionsOnCam(url, camName):
    global config

    newFrame = None
    detector  = detect2.detector(camName, config.get('model', 'protocol'), config.get('model', 'model'))
    cam = webcam.threadCamReader(url)
    cam.start()
    sender = imagezmq.ImageSender(connect_to='tcp://*:5551' ,block = False)    
    readFrameID = None
    while True:
        time.sleep(1/10000)
        frame, frameID    = cam.read()
        if frame is None or frameID is None or readFrameID == frameID:
            continue
        
        readFrameID = frameID
        newFrame    = detector.detect(frame, int(config.get(camName, 'fps')))
                
        if newFrame is not None:
            sender.send_image(camName, newFrame)

def readToWeb():
    while True:
        outFrame = np.concatenate(list(output.values()))
        yield prepMultipart(outFrame)

def montage():
    global output
    receiver = imagezmq.ImageHub(open_port='tcp://localhost:5551', block = False)
    receiver.connect(open_port = 'tcp://127.0.0.1:5551')
    while True:
         time.sleep(1/10000)   
         camName, frame = receiver.recv_image()
         (h, w) = frame.shape[:2]
         frame = cv2.copyMakeBorder(frame, round((w - h) / 2), round((w - h) / 2), 0, 0, cv2.BORDER_CONSTANT,  (0,0,0,0))
         output[camName] = cv2.resize(frame, (640, 480))

@app.route('/')
def application():
    return Response(readToWeb(), mimetype='multipart/x-mixed-replace; boundary=frame')

def startWeb():
    thread = threading.Thread(target=montage)
    thread.daemon = True
    thread.start()    
    run_simple("0.0.0.0", 4000, app)

if __name__ == '__main__':
    p = Process(target=runDetectionsOnCam, args=(cam, camName))
    p.start()
    p1 = Process(target=startWeb, args=())
    p1.start()
    p1.join()

