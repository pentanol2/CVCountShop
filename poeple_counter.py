# core python imports
import time
import argparse

# third party imports
import dlib
import imutils
import numpy as np
import cv2 as cv

from imutils.video import VideoStream
from imutils.video import FPS # helps calculate the estimated fps

# local imports
from track.centroid_tracker import CentroidTracker
from track.trackable_object import TrackableObject


# parsing command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p','--prototxt',required=True,help='path to Caffe prototxt deploy file.')
ap.add_argument('-m','--model',required=True,help='path to Caffe pretrained model')
ap.add_argument('-i','--input',type=str,help='[optional] path to the input video file')
ap.add_argument('-o','--output',type=str,help='[optional] path to output video file')
ap.add_argument('-c','--confidence',type=float,default=0.4,help='confidence threshold to filter weak detections')
ap.add_argument('-s','--skip',type=int,default=30,help='# frames to skip before relaunching the detection')

args = vars(ap.parse_args())

# classes for the SSD model / detector
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"
]


# loading model from the disk
model = cv.dnn.readNetFromCaffe(args['prototxt'],args['model'])

# in case an input video is not supplied
if not args.get('input',False):
    print('[INFO] starting video stream ...')
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else: # ref to video file
    print('[INFO] opening video file ...')
    vs = cv.VideoCapture(args['input'])

# init vid writer
vwriter = None


# readed video frame dimensions
w = None
h = None

# centroid tracker set
ctracker = CentroidTracker(maxDisappeared= 40,maxDistance=40)
# corr filers trackers
trackers = []
# set of trackable objects
trackable_objects = {}

# number of frames processed too far
total_frames = 0
# number of objects that moved up
total_up = 0
# number of objects that moved down
total_down = 0

# init number of frames per second

fps = FPS().start()

# video stream processing loop
while True:
    # we grab the frame to handle
    frame = vs.read()
    frame = frame[1] if args.get('input',False) else frame

    # in case we cant get the video frame while input is set up we breal the loop / End of the video
    if args['input'] is not None and frame is None:
        break

    # we resize the frame to have a max width of 500
    frame = imutils.resize(frame,width=500)
    rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    # set frame width and height
    if h is None or w is None:
        (h,w) = frame.shape[:2]

    # set up video writer if output flag is set
    if args['output'] is not None and vwriter is None:
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        vwriter = cv.VideoWriter(args['output'],fourcc,30,(w,h),True)

    # detection phase
    status = 'waiting'
    # list of bounding boxes
    bbs = []

    # we check if object detection should be run in this frame
    if total_frames % args['skip'] == 0:
        status = 'detecting'
        # print(status)
        trackers = []

        # conv frame to blob
        blob = cv.dnn.blobFromImage(frame,0.007843, (w, h), 127.5)
        # pass frame through network
        model.setInput(blob)
        detections = model.forward()

        # we loop over the detected objects in this current frame
        for d in range(0,detections.shape[2]):
            # get the confidence score for the bounding box 'd'
            confidence = detections[0,0,d,2]

            # compare it to confidence threshold input by the user
            if confidence > args['confidence']:
                # we get the class index

                class_index = int(detections[0,0,d,1])

                if CLASSES[class_index] != 'person':
                    continue
                print('Confidence score {} class {} '.format(confidence, CLASSES[class_index]))
                # we calculate the on image bbox coordinates for a specific object
                bbox = detections[0,0,d,3:7] * np.array([w,h,w,h])
                # we convert bbox coordinates to intergers
                (Ax,Ay,Bx,By) = bbox.astype('int')

                # create a dlib correlation tracker for the current object / bbox
                tracker = dlib.correlation_tracker()
                # set up a bbox to track for the tracker
                rect = dlib.rectangle(Ax,Ay,Bx,By)
                # start to track the rect / bbox on the rgb image / frame
                tracker.start_track(rgb,rect)

                # add the tracker to the global list of trackers
                trackers.append(tracker)

    # leaving detection to tracking mode
    else:
        # looping over all object and update each bb's coordinates
        for tracker in trackers:
            status = 'tracking'
            # update the object's current coordinates
            tracker.update(rgb)
            # get the new position
            pos = tracker.get_position()

            # unpack new position coordinates
            Ax = int(pos.left())
            Ay = int(pos.top())
            Bx = int(pos.right())
            By = int(pos.bottom())

            # add the new position for the list of bounding boxes corresponding to the current frame
            bbs.append((Ax,Ay,Bx,By))
            # print(bbs)
    # drawing the counting line
    cv.line(frame,(0,h//2),(w,h//2),(255,0,255),2)
    # get the list of centroids
    objects = ctracker.update(bbs)

    # loop over the tracked objects
    for (objectId,centroid) in objects.items():
        # we check if the object exists in the list of trackable objects
        trackable = trackable_objects.get(objectId,None)
        # print(centroid)

        # if the object is not in the trackable object list
        if trackable is None:
            # create a trackable object for the object with the centroid in hand
            trackable = TrackableObject(objectId,centroid)
        # and if the object is in the list of tracked trackable objects
        else:
            # let's specify the direction of the moving object
            historical_centroids = trackable.centroids
            # print(historical_centroids)
            # calculate the direction
            position_mean = np.mean([c[1] for c in historical_centroids])
            direction = centroid[1] - position_mean
            # add the newly calculated centroid to the list of historical centroids of the current tackable object
            trackable.centroids.append(centroid)

            # if the object is not counted just count it
            if not trackable.counted:
                # check for up
                if direction > 0 and centroid[1] > h//2:
                    trackable.counted = True
                    total_up += 1
                # check for down
                if direction < 0 and centroid[1] < h//2:
                    trackable.counted = True
                    total_down += 1

        # update / store trackable object in the dictionary
        trackable_objects[objectId] = trackable
        # visualization
        # id and centroid of the object
        text = 'ID {}'.format(objectId)
        cv.putText(frame,text,(centroid[0]-10,centroid[1]-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv.circle(frame,(centroid[0],centroid[1]),4,(0,255,0),-1)

        # info to display on the frame
        info = [('up',total_up),('down',total_down),('currenlty',status)]

        for (l,(key,value)) in enumerate(info):
            text = '{} -> {}'.format(key,value)
            # print(text)
            cv.putText(frame,text,(20,(l*20)+10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        # print('---------------------')
    # write frame to vid
    if vwriter is not None:
        vwriter.write(frame)

    # displaying the output frame
    cv.imshow('See how we can count things man :)',frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('b'):
        break

    # updating the total of frames we processed so far
    total_frames += 1
    fps.update()

# garbage collection and cleanup
# stop counting:
fps.stop()
print('[INFO] total tracking time {:.2f}'.format(fps.elapsed()))
print('[INFO] approximate processed frames per second {:.2f}'.format(fps.fps()))

# collect vwriter if we have one
if vwriter is not None:
    vwriter.release()

# stop video stream from camera in case we re not using an input video
if not args.get('input',False):
    vs.stop()
else:
    vs.release()

# we close all windows
cv.destroyAllWindows()








