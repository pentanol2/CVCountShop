from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self,maxDisappeared=50,maxDistance = 50):
        # we establish the next unique object id
        self.nextObjectId = 0
        # map object id to tracking metadata
        self.objects = OrderedDict() # key: object_id value: centroid_coordinates / the current ones
        self.disappeared = OrderedDict() # key: object_id value: number of frames an object is marked as lost

        # maxframes needed to deregister an object
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self,centroid):
        # add new object to be tracked
        self.objects[self.nextObjectId] = centroid
        self.disappeared[self.nextObjectId] = 0
        self.nextObjectId += 1

    def deregister(self,objectId):
        # simply done by deleting the objects from objects and disappeared dicts
        del self.objects[objectId]
        del self.disappeared[objectId]

    def update(self,bboxes):  # the bboxes is the output of the object detection algorithm
        # hold the logic of the centroid tracker
        if len(bboxes) == 0: # if there bounding boxes to track / no object detected
            for objectId in self.disappeared.keys():
                self.disappeared[objectId] += 1
                # print('Updated : ',self.disappeared[objectId])
                if self.disappeared[objectId] > self.maxDisappeared:
                    self.deregister(objectId)
            return self.objects
        # in case we bbs are detected
        # we intialize an array of centroids
        newCentroids = np.zeros((len(bboxes),2),dtype=int)

        for (bb,(Ax,Ay,Bx,By)) in enumerate(bboxes):
            Ox = int((Ax+Bx)/2)
            Oy = int((Ay+By)/2)
            newCentroids[bb] = (Ox,Oy)

        # if we're tracking no objects register the detected ones
        if len(self.objects) == 0:
            for c in newCentroids:
                self.register(c)
        else:
            # we need to update the objects coordinates
            objectIds = list(self.objects.keys())
            # get all objects' centroids
            objectCentroids = list(self.objects.values())
            # calc the distance % existing object centroids and input centroids
            dist_updates = dist.cdist(np.array(objectCentroids),newCentroids)
            # we find the rows of mind distances
            rows = dist_updates.min(axis=1).argsort()
            # we find the cols of min distances corresponding to each object
            cols = dist_updates.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # updating existing object with the new input centroids
            # looping over distance combinations
            # if row or column is used just continue
            for (r,c) in zip(rows,cols):
                if r in usedRows or c in usedCols:
                    continue

                if dist_updates[r,c] > self.maxDistance:
                    continue
                # if not
                # set the id of the object to be updated
                objectId = objectIds[r]
                # update the existing object with the new input centroid
                self.objects[objectId] = newCentroids[c]
                # reset the disappeared counter to zero
                self.disappeared[objectId] = 0

                # mark the object as updated by setting the corresponding row col as used
                usedRows.add(r)
                usedCols.add(c)

            # we check for the unused rows/columns , objectCentroid/inputCentroid
            unusedRows = set(range(0,dist_updates.shape[0])).difference(usedRows)
            unusedCols = set(range(0,dist_updates.shape[1])).difference(usedCols)

            # in case nrows >= ncols there is a possiblity the existing objects were lost
            if dist_updates.shape[0] >= dist_updates.shape[1]:
                # loop over the unused rows
                for r in unusedRows:
                    # get the objectId
                    objectId = objectIds[r]
                    # set the object as disappeared one time more
                    self.disappeared[objectId] += 1

                    # if the number of consecutive disappearence is more than our disappearence threshold
                    if self.disappeared[objectId] > self.maxDisappeared:
                        self.deregister(objectId)
            else:
            # in case the number of centroids is less than the number of input centroids then we have new centroids to register
                for c in unusedCols:
                    self.register(newCentroids[c])

            # we return the final updated object list
        return self.objects




