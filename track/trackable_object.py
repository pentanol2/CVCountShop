class TrackableObject:
    # we intialize the object tracker
    def __init__(self,objectId,centroid):
        # the object to be tracked has an Id
        self.objectId = objectId
        # has a record of historical centroid positions
        self.centroids = [centroid]

        # flag to indicate if the object is counted or not is needed
        self.counted = False # set to false by default until counted
