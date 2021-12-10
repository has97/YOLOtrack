from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxdisapp=45):

		self.nextOBID = 0 # unique object id
		self.objecdict = OrderedDict()# mapping of object id to centroid
		self.disapp = OrderedDict() # number of times a particular object is marked as disapp from the frame
		self.maxdisapp = maxdisapp # maximum number of times an object can be marked as disapp

	def registering(self, centroid):
		self.nextOBID += 1 # updating next object_id for the next object
		self.disapp[self.nextOBID] = 0 # initializing the disapp variable of object as zero
		self.objecdict[self.nextOBID] = centroid # registering the object with its centroid 

	def deregistering(self, objectID):
		del self.objecdict[objectID] # deleting the object from mapping list for deregistering
		del self.disapp[objectID] # deleting the object from disappearing list for deregistering

	def update(self, rects):

		if len(rects) == 0: # no detection
		
			for objectID in self.disapp.keys(): #marking the tracked object as disapp by incrementing
				self.disapp[objectID] += 1

				if self.disapp[objectID] > self.maxdisapp: # deregister the object because it went beyond the threshold
					self.deregistering(objectID)

			return self.objecdict

		# This holds the centroids of the detections
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# looping over the bounding boxes
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY) # centroid of rectangle

		if len(self.objecdict) == 0: # no objecdict are registered yet
			for i in range(0, len(inputCentroids)):
				self.registering(inputCentroids[i])

		else:  # objecdict are registered 
			
			objectIDs = list(self.objecdict.keys()) # object id list
			objectCentroids = list(self.objecdict.values()) # object centroid list

			Distance = dist.cdist(np.array(objectCentroids), inputCentroids)# matrix with pair wise distance of object and input centroid

			# obtaining the minimum value along the row and then sort this values
			rows = Distance.min(axis=1).argsort()

			# next, we perform a similar process on the columns by using previously computed row index
			cols = Distance.argmin(axis=1)[rows]

			# this is to update the object : either registering, update or deregistering
			usedObjectRow = set()
			usedObjectCols = set()

	
			for (row, col) in zip(rows, cols):
				
				if row in usedObjectRow or col in usedObjectCols: #multiple detection for an object
					continue

				objectID = objectIDs[row] # assigning object id to that centroid
				self.objecdict[objectID] = inputCentroids[col]# assigning new centroid
				self.disapp[objectID] = 0 # updating disapp status

				# get the list of 
				usedObjectRow.add(row)
				usedObjectCols.add(col)

			# some objecdict have disapp, That is being captured in this 
			unusedRowobj = set(range(0, Distance.shape[0])).difference(usedObjectRow)
			unusedColobj = set(range(0, Distance.shape[1])).difference(usedObjectCols)

			# Some objecdict are missing
			if Distance.shape[0] >= Distance.shape[1]:

				for row in unusedRowobj:#loop over the disapp status
			        
					objectID = objectIDs[row]
					self.disapp[objectID] += 1 #updating the disappearence status

					if self.disapp[objectID] > self.maxdisapp: # if greater than threshold deregister the object
						self.deregisteing(objectID)

			else:
				for col in unusedColobj: # loop over unregistered object
					self.registering(inputCentroids[col])

		# return the set of trackable objecdict
		return self.objecdict