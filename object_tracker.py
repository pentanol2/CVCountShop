# core python import
import time

# third party imports
import imutils
import numpy as np
import cv2 as cv
from imutils.video import VideoStream

# local imports
from track.centroid_tracker import CentroidTracker


model_path = ['models/cd_deploy.prototxt','models/res10_300x300_ssd_iter_140000.caffemodel']
# model_path = ['models/MobileNetSSD_deploy.prototxt','models/MobileNetSSD_deploy.caffemodel']

# create centroid tracker

tracker = CentroidTracker()
(h,w) = (None,None)

# loading model from the dist
print('[INFO] Loading model from disk')
model = cv.dnn.readNetFromCaffe(model_path[0],model_path[1])


# we initialize a video stream
print('[INFO] Starting video stream')
vs = VideoStream(src=0).start()
time.sleep(2.0)

# we loop over our video frames
while True:
    # we use read() fn to read the next frame
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # set frame dimensions
    if h is None or w is None:
        (h,w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame,1.0,(w,h),(104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    rects = []
    for c in range(0,detections.shape[2]):
        if detections[0,0,c,2] > 0.85:
            bb = detections[0,0,c,3:7] * np.array([w,h,w,h])
            bb = bb.astype('int')
            rects.append(bb)

            # now we draw the bb before visualizing it
            (Ax,Ay,Bx,By) = bb
            cv.rectangle(frame,(Ax,Ay),(Bx,By),(0,255,0),2)

    # we can now update the tracker with the set of bb detected
    tracker.update(rects)

    # we scan the tracked objects
    for (objectId,centroid) in tracker.objects.items():
        # drawing object id and corresponding centroids
        text = 'ID {}'.format(objectId)

        cv.putText(frame,text,(centroid[0]-10,centroid[1]-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv.circle(frame,(centroid[0],centroid[1]),4,(0,255,0),-1)

    cv.imshow('See what we can detect :)',frame)
    key = cv.waitKey(1) or 0xFF

    # we break the the loop when the 'b' key is pressed
    if key == 'b':
        break

cv.destroyAllWindows()
vs.stop()
