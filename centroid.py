from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=45):

		self.nextObjectID = 0 # unique object id
		self.objects = OrderedDict()# mapping of object id to centroid
		self.disappeared = OrderedDict() # number of times a particular object is marked as disappeared from the frame
		self.maxDisappeared = maxDisappeared # maximum number of times an object can be marked as disappeared

	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid # registering the object with its centroid 
		self.disappeared[self.nextObjectID] = 0 # initializing the disappeared variable of object as zero
		self.nextObjectID += 1 # updating next object_id for the next object

	def deregister(self, objectID):
		del self.objects[objectID] # deleting the object from mapping list for deregistering
		del self.disappeared[objectID] # deleting the object from disappearing list for deregistering

	def update(self, rects):

		if len(rects) == 0: # no detection
		
			for objectID in self.disappeared.keys(): #marking the tracked object as disappeared by incrementing
				self.disappeared[objectID] += 1

				if self.disappeared[objectID] > self.maxDisappeared: # deregister the object because it went beyond the threshold
					self.deregister(objectID)

			return self.objects

		# This holds the centroids of the detections
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# looping over the bounding boxes
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY) # centroid of rectangle

		if len(self.objects) == 0: # no objects are registered yet
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		else:  # objects are registered 
			
			objectIDs = list(self.objects.keys()) # object id list
			objectCentroids = list(self.objects.values()) # object centroid list

			D = dist.cdist(np.array(objectCentroids), inputCentroids)# matrix with pair wise distance of object and input centroid

			# obtaining the minimum value along the row and then sort this values
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by using previously computed row index
			cols = D.argmin(axis=1)[rows]

			# this is to update the object : either registering, update or deregistering
			usedRows = set()
			usedCols = set()

	
			for (row, col) in zip(rows, cols):
				
				if row in usedRows or col in usedCols: #multiple detection for an object
					continue

				objectID = objectIDs[row] # assigning object id to that centroid
				self.objects[objectID] = inputCentroids[col]# assigning new centroid
				self.disappeared[objectID] = 0 # updating disappeared status

				# get the list of 
				usedRows.add(row)
				usedCols.add(col)

			# some objects have disappeared, That is being captured in this 
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# Some objects are missing
			if D.shape[0] >= D.shape[1]:

				for row in unusedRows:#loop over the disappeared status
			        
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1 #updating the disappearence status

					if self.disappeared[objectID] > self.maxDisappeared: # if greater than threshold deregister the object
						self.deregister(objectID)

			else:
				for col in unusedCols: # loop over unregistered object
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects