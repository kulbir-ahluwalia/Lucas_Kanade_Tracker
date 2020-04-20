import glob
import cv2
import numpy as np


def getHomogenCoords(x_range,y_range):
	a = (x_range[1] - x_range[0]) + 1
	b = (y_range[1] - y_range[0]) + 1
	coordinates = np.zeros((3, a*b))
	count = 0
	for y in range(y_range[0], y_range[1] + 1):
		for x in range(x_range[0], x_range[1] + 1):
			coordinates[0, count] = x
			coordinates[1, count] = y
			coordinates[2, count] = 1
			count += 1

	return coordinates


def warpTF(tempCoords, p, x_range, y_range):
	x1, x2 = x_range
	y1, y2 = y_range
	array = np.array([[x1,x1,x2,x2],[y1,y2,y2,y1],[1,1,1,1]])

	affineWarpMat = np.zeros((2,3))
	count = 0
	for i in range(3):
		for j in range(2):
			affineWarpMat[j,i] = p[count,0] 
			count += 1 
	affineWarpMat[0,0] += 1
	affineWarpMat[1,1] += 1
	warpedVertices = np.dot(affineWarpMat,array)
	new = np.dot(affineWarpMat,tempCoords)
	new = new.astype(int)

	return new_coordinates,warpedVertices


def getPixelArray(image,coordinates):
	img_array = np.zeros((1,coordinates.shape[1]))
	img_array[0,:] = image[coordinates[1,:],coordinates[0,:]]  

	return img_array


def getTemplateArray(template, x_range, y_range):
	tempCoords = getHomogenCoords(x_range,y_range)
	p = np.array([[0,0,0,0,0,0]]).T
	newCoords, newVer = warpTF(tempCoords, p ,x_range, y_range)
	array = getPixelArray(template, new_coordinates)
	
	return array


def isFeasible(coorinate_array, img) :
	min_=np.amin(coorinate_array,axis=1)
	max_=np.amax(coorinate_array,axis=1)

	if min_[0] < 0 or max_[0] >= img.shape[1] or min_[1] < 0 or max_[1] >= img.shape[0]:
		return False
	else:
		return True

   
def getSteepestDescent(sobelx,sobely,new_coordinates,old_coordinates):

	gradX = getPixelArray(sobelx,new_coordinates)
	gradY = getPixelArray(sobely,new_coordinates)
	img1 = gradX*old_coordinates[0,:]
	img2 = gradY*old_coordinates[0,:]
	img3 = gradX*old_coordinates[1,:]
	img4 = gradY*old_coordinates[1,:]
	steepest = np.vstack((img1, img2, img3, img4, gradX, gradY)).T

	return steepest


def affineLKTracker(tempCoords, temp, gray_image, x_range, y_range, p, sobelx, sobely):

	warpedTemp, warpedVertices = warpTF(tempCoords, p, x_range, y_range)

	if isFeasible(warpedTemp,gray_image):
		feasible = True
		img_arr = getPixelArray(gray_image,warpedTemp)

		error = temp - img_arr
		steepest = getSteepestDescent(sobelx, sobely, warpedTemp, tempCoords)
		SDProd = np.dot(steepest.T, error.T)
		Hess = np.dot(steepest.T, steepest)
		Hinv = np.linalg.pinv(Hess)
		deltaP = np.dot(Hinv, SDProd)
		p_norm = np.linalg.norm(deltaP)
		p = np.reshape(p,(6,1))
		p = p + deltaP

	else:
		feasible = False
		deltaP = np.array([[0,0,0,0,0,0]]).T
		p_norm = np.linalg.norm(deltaP)

	return p, deltaP, p_norm, warpedTemp, warpedVertices, feasible


#dataset = "Bolt2"
#dataset = "Car4"
dataset = "DragonBaby"
path = dataset + "/img/*.jpg"
outputPath = "output/" + dataset + "/"

dataDict = {}
dataDict["Bolt2"] = {"x_range": [265, 308], "y_range": [80, 145], "threshold": 0.0159, "ksize": 5}
dataDict["Car4"] = {"x_range": [65, 180], "y_range": [45, 140], "threshold": 0.01, "ksize": 5}
dataDict["DragonBaby"] = {"x_range": [90, 240], "y_range": [70, 300], "threshold": 0.078, "ksize": 3}

first = cv2.imread(dataset + "/img/0001.jpg")  
first = cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)
firstMean = np.mean(first)
threshold = dataDict[dataset]["threshold"]

x_range = dataDict[dataset]["x_range"]
y_range = dataDict[dataset]["y_range"]

# cv2.line(first, (x_range[0], y_range[0]), (x_range[1], y_range[0]), (0, 255, 0), 4)
# cv2.line(first, (x_range[1], y_range[0]), (x_range[1], y_range[1]), (0, 255, 0), 4)
# cv2.line(first, (x_range[1], y_range[1]), (x_range[0], y_range[1]), (0, 255, 0), 4)
# cv2.line(first, (x_range[0], y_range[0]), (x_range[0], y_range[1]), (0, 255, 0), 4)

# cv2.imshow('first', first)
# if cv2.waitKey(0) & 0xff == 27:	
# 	cv2.destroyAllWindows()

temp = getTemplateArray(first,x_range,y_range)
tempCoords = getHomogenCoords(x_range,y_range)

params = np.array([[0,0,0,0,0,0]]).T
count = 0

for img in sorted(glob.glob(path)):
	        
	image = cv2.imread(img)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	grayMean = np.mean(gray)
	gray = (gray*((firstMean/grayMean))).astype(float)
	sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=dataDict[dataset]["ksize"])
	sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=dataDict[dataset]["ksize"])

	while True:
		params, deltaP, p_norm, warpedTemp, new_vertex, feasible = affineLKTracker(tempCoords,temp,gray,x_range,y_range,params,sobelx,sobely)
		if (p_norm <= threshold) or (feasible == False):
			break

	image = cv2.polylines(image, np.int32([new_vertex.T]), 1, (100,100,255), 2)
	cv2.imshow('output',image)
	cv2.imwrite(outputPath + str(count) + '.jpg', image)    

	count += 1
	if cv2.waitKey(1) & 0xff == 27:	
	 	break
	 	cv2.destroyAllWindows()
	
	
cv2.destroyAllWindows()