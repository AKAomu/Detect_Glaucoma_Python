## import ##
## openCV ##
import cv2 
import numpy as np
import pandas as pd
import pylab as pl
import os
import glob
import ntpath
import xlrd
from matplotlib import pyplot as plt
from PIL import Image

## PyWavelet ##
import pywt

from skimage import io
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif ,SelectKBest, f_regression, SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

## Scipy ##
from scipy import stats


def loadImagesFromFolder(folder):
	images = []
	fileName = []
	for filename in glob.glob(folder):
		# img = np.float32(cv2.imread(filename))
		img = io.imread(filename)
		if img is not None:
			images.append(img)
			# name = ntpath.basename(filename).split("_")
			# fileName.append(name[0])
			fileName.append(ntpath.basename(filename))
			# print(filename)
	return images , fileName


def img_to_matrix(img, verbose=False):
	"""
	takes a filename and turns it into a numpy array of RGB pixels
	"""
	widthBase = 700.0 / img.shape[1]
	dim = (700, int(img.shape[0] * widthBase))
	img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	# STANDARD_SIZE = (300, 167)	
	# img = Image.open(filename)
	# if verbose==True:
		# print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
	# img = img.resize(STANDARD_SIZE)
	# img = list(img)
	# img = map(list, img)
	img = np.array(img)
	# img2 = np.asarray(img)
	
	# print (img)
	# plt.imshow(img)
	# plt.show()
	return img

def flatten_image(data):
	"""
	takes in an (m, n) numpy array and flattens it 
	into an array of shape (1, m * n)
	"""
	# print(img.shape[0])
	# print(img.shape[1])
	# cv2.imshow('image',img)
	# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# plt.imshow(gray)
	# plt.show()
	# X_normalized = preprocessing.normalize(img, norm='l2')
	
	# s = img.shape[0] * img.shape[1]
	# img_wide = img.reshape((1, s,-1))	
	# img_wide = np.rollaxis(X_normalized, axis=1, start=0)
	# plt.imshow(img_wide[0])
	# plt.show()
	# print(X_normalized)
	nsamples, nx, ny = data.shape
	d2_train_dataset = data.reshape((nsamples,nx*ny))
	return d2_train_dataset
	
def svmTraining(img) :
	cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

	# First half is trainData, remaining is testData
	train_cells = [ i[:50] for i in cells ]
	test_cells = [ i[50:] for i in cells]
	
	######     Now training      ########################

	deskewed = [map(deskew,row) for row in train_cells]
	hogdata = [map(hog,row) for row in deskewed]
	trainData = np.float32(hogdata).reshape(-1,64)
	responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])

	svm = cv2.SVM()
	svm.train(trainData,responses, params=svm_params)
	svm.save('svm_data.dat')

	######     Now testing      ########################

	deskewed = [map(deskew,row) for row in test_cells]
	hogdata = [map(hog,row) for row in deskewed]
	testData = np.float32(hogdata).reshape(-1,bin_n*4)
	result = svm.predict_all(testData)

	#######   Check Accuracy   ########################
	mask = result==responses
	correct = np.count_nonzero(mask)
	print (correct*100.0/result.size)	
	
def cornerDetect(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(gray,20,0.01,10)
	corners = np.int0(corners)

	for i in corners:
		x,y = i.ravel()
		cv2.circle(img,(x,y),3,255,-1)

	# plt.imshow(img)
	# plt.show()
	# cv2.waitKey(1)
	return corners
	
def getTrainingData():
	address = "G:/My Drive/ICTES/Thesis/Code/od_roi/DataMining/training"
	labels = []
	trainingData = []
	for items in os.listdir(address):
		## extracts labels
		name = address + "/" + items
		print (items)
		for it in os.listdir(name):
			path = name + "/" + it
			print (path)
			img = cv.imread(path, cv.CV_LOAD_IMAGE_GRAYSCALE)
			d = np.array(img, dtype = np.float32)
			q = d.flatten()
			trainingData.append(q)
			labels.append(items)
			######DEBUG######

			#cv.namedWindow(path,cv.WINDOW_NORMAL)
			#cv.imshow(path,img)

	return trainingData, labels

# def pca(X):
	# # Principal Component Analysis
	# # input: X, matrix with training data as flattened arrays in rows
	# # return: projection matrix (with important dimensions first),
	# # variance and mean

	# #get dimensions
	# num_data,dim = X.shape
	
	# #center data
	# mean_X = X.mean(axis=0)
	# for i in range(num_data):
		# X[i] -= mean_X
	# X = X.reshape(-1, 3)

	# if dim>100:
		# print ('PCA - compact trick used')
		# M = np.dot(X,X.T) #covariance matrix
		# e,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
		# tmp = np.dot(X.T,EV).T #this is the compact trick
		# V = tmp[::-1] #reverse since last eigenvectors are the ones we want
		# # print(EV)
		# S = np.sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
	# else:
		# print ('PCA - SVD used')
		# U,S,V = np.linalg.svd(X)
		# V = V[:num_data] #only makes sense to return the first num_data

	# #return the projection matrix, the variance and the mean
	# return V,S,mean_X
	
def PCAs(X_train, X_test, components):		
	# pca = PCA(n_components=components, svd_solver='randomized')
	pca = PCA(n_components=components)
	train_x = pca.fit_transform(X_train)
	test_x = pca.transform(X_test)	
	print("pca coeffs")
	print (train_x)
	print (test_x)
	# df = pd.DataFrame({"x": test_x[:, 0], "y": test_x[:, 1], "label":np.where(Y_test==1, "Glaucoma test", "Normal test")})
	# colors = ["blue", "green"]
	# for label, color in zip(df['label'].unique(), colors):
	# mask = df['label']==label
	# pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
	# pl.legend()
	# pl.show()
	# io.imshow(train_x)
	# plt.show()
	
	return train_x , test_x

def KNN(train_x, test_x, train_y, test_y, kernels):
	target_names = ['Normal', 'Glaucoma']
	knn = KNeighborsClassifier(kernels)
	knn.fit(train_x, train_y)
	y_pred = knn.predict(test_x)
	print("KNN")
	# print(pd.crosstab(test_y, knn.predict(test_x), rownames=["Actual"], colnames=["Predicted"]))
	print(classification_report(test_y, y_pred,target_names=target_names))
	print("label test_x")
	print(test_y)
	print("predict test_x")
	print(knn.predict(test_x))
	
def SVM(train_x, test_x, train_y, test_y, kernels):
	# ANOVA SVM-C
	# 1) anova filter, take 3 best ranked features
	anova_filter = SelectKBest(f_regression, k=kernels)
	# 2) svm
	clf = svm.SVC(kernel='linear')

	anova_svm = make_pipeline(anova_filter, clf)
	anova_svm.fit(train_x, train_y)
	# clf.fit(train_x, Y_train)
	y_pred = anova_svm.predict(test_x)
	target_names = ['Normal', 'Glaucoma']
	print("SVM")
	print(classification_report(test_y, y_pred,target_names=target_names))
	print("label test_x")
	print(test_y)
	print("predict test_x")
	print(y_pred)
	
def Wavelet(data):		
	coeffs = pywt.dwt2(data, 'sym3')
	cA, (cH, cV, cD) = coeffs
	
	cA, cD = pywt.dwt(data, 'sym3')	
	
	# wp = pywt.WaveletPacket2D(data=train_x, wavelet='sym3', mode='symmetric')
	# print(wp['va'].data)	
	# print([node.path for node in wp.get_level(2)])
	re = pywt.idwt2(coeffs, 'sym3')
	print(re)	

	## plot ##
	# plt.figure(1)
	# plt.subplot(211)	
	# plt.plot(data)
	# plt.title('input')
	# plt.subplot(212)
	# plt.plot(re)
	# plt.title('reconstructs')
	# plt.show()
	# plt.imshow(re)
	# plt.show()
	# print(train_cA)
	# print(train_cV)
	# print(train_cD)	
	return re 
	
# def Wavelet_avg(data,ans):
	# coeffs = pywt.dwt2(data, 'sym3')
	# cA, (cH, cV, cD) = coeffs
	
	# # cA, cD = pywt.dwt(data, 'sym3')	
	
	# # wp = pywt.WaveletPacket2D(data=train_x, wavelet='sym3', mode='symmetric')
	# # print(wp['va'].data)	
	# # print([node.path for node in wp.get_level(2)])
	# # re = pywt.idwt2(coeffs, 'sym3')
	# # print(re)
	
	# # print(cH.shape)
	# # print(data.shape)
	# p , q = data.shape
	# avg_cH = sum(abs(cH))/p*q
	# avg_cV = sum(abs(cV))/p*q
	# avg_cD = sum(abs(cD))/p*q

	# enegy =  pow(sum(abs(cV)),2)/pow(p*q,2)

	# # print("avg_cH")
	# # # print(avg_cH)
	# # print(avg_cH.shape)

	# # print("avg_cV")
	# # # print(avg_cV)
	# # print(avg_cV.shape)

	# # print("enegy")
	# # # print(enegy)	
	# # print(enegy.shape)	
	
	# ans_return = {
	# 'cH': avg_cH,
	# 'cV': avg_cV,
	# 'cD': avg_cD,
	# 'enegy': enegy}[ans]
	
	# return ans_return

def Wavelet_avg(data):
	## db3 ##
	sum_cH_db3 = 0
	sum_cV_db3 = 0
	sum_cD_db3 = 0
	
	coeffs_db3 = pywt.dwt2(data, 'db3')
	cA_db3, (cH_db3, cV_db3, cD_db3) = coeffs_db3
	
	p , q = cH_db3.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_db3 += abs(cH_db3[x][y])
			sum_cV_db3 += abs(cV_db3[x][y])
			sum_cD_db3 += abs(cD_db3[x][y])
	
	avg_cH_db3 = sum_cH_db3/p*q
	avg_cV_db3 = sum_cV_db3/p*q
	avg_cD_db3 = sum_cD_db3/p*q
	
	enegy_cH_db3 =  pow(sum_cH_db3,2)/pow(p*q,2)
	enegy_cV_db3 =  pow(sum_cV_db3,2)/pow(p*q,2)
	enegy_cD_db3 =  pow(sum_cD_db3,2)/pow(p*q,2)
	
	## sym3 ##
	sum_cH_sym3 = 0
	sum_cV_sym3 = 0
	sum_cD_sym3 = 0
	
	coeffs_sym3 = pywt.dwt2(data, 'sym3')
	cA_sym3, (cH_sym3, cV_sym3, cD_sym3) = coeffs_sym3
	
	p , q = cH_sym3.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_sym3 += abs(cH_sym3[x][y])
			sum_cV_sym3 += abs(cV_sym3[x][y])
			sum_cD_sym3 += abs(cD_sym3[x][y])
	# print(sum_cH)
	# print(sum_cV)
	# print(sum_cD)
	avg_cH_sym3 = sum_cH_sym3/p*q
	avg_cV_sym3 = sum_cV_sym3/p*q
	avg_cD_sym3 = sum_cD_sym3/p*q
	
	enegy_cH_sym3 =  pow(sum_cH_sym3,2)/pow(p*q,2)
	enegy_cV_sym3 =  pow(sum_cV_sym3,2)/pow(p*q,2)
	enegy_cD_sym3 =  pow(sum_cD_sym3,2)/pow(p*q,2)
	
	## rbio3.3 ##
	sum_cH_rbio33 = 0
	sum_cV_rbio33 = 0
	sum_cD_rbio33 = 0
	
	coeffs_rbio33 = pywt.dwt2(data, 'rbio3.3')
	cA_rbio33, (cH_rbio33, cV_rbio33, cD_rbio33) = coeffs_rbio33
	
	p , q = cH_rbio33.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_rbio33 += abs(cH_rbio33[x][y])
			sum_cV_rbio33 += abs(cV_rbio33[x][y])
			sum_cD_rbio33 += abs(cD_rbio33[x][y])
	
	avg_cH_rbio33 = sum_cH_rbio33/p*q
	avg_cV_rbio33 = sum_cV_rbio33/p*q
	avg_cD_rbio33 = sum_cD_rbio33/p*q
	
	enegy_cH_rbio33 =  pow(sum_cH_rbio33,2)/pow(p*q,2)
	enegy_cV_rbio33 =  pow(sum_cV_rbio33,2)/pow(p*q,2)
	enegy_cD_rbio33 =  pow(sum_cD_rbio33,2)/pow(p*q,2)
	
	## rbio3.5 ##
	sum_cH_rbio35 = 0
	sum_cV_rbio35 = 0
	sum_cD_rbio35 = 0
	
	coeffs_rbio35 = pywt.dwt2(data, 'rbio3.5')
	cA_rbio35, (cH_rbio35, cV_rbio35, cD_rbio35) = coeffs_rbio35
	
	p , q = cH_rbio35.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_rbio35 += abs(cH_rbio35[x][y])
			sum_cV_rbio35 += abs(cV_rbio35[x][y])
			sum_cD_rbio35 += abs(cD_rbio35[x][y])
	
	avg_cH_rbio35 = sum_cH_rbio35/p*q
	avg_cV_rbio35 = sum_cV_rbio35/p*q
	avg_cD_rbio35 = sum_cD_rbio35/p*q
	
	enegy_cH_rbio35 =  pow(sum_cH_rbio35,2)/pow(p*q,2)
	enegy_cV_rbio35 =  pow(sum_cV_rbio35,2)/pow(p*q,2)
	enegy_cD_rbio35 =  pow(sum_cD_rbio35,2)/pow(p*q,2)
	
	## rbio3.7 ##
	sum_cH_rbio37 = 0
	sum_cV_rbio37 = 0
	sum_cD_rbio37 = 0
	
	coeffs_rbio37 = pywt.dwt2(data, 'rbio3.7')
	cA_rbio37, (cH_rbio37, cV_rbio37, cD_rbio37) = coeffs_rbio35
	
	p , q = cH_rbio37.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_rbio37 += abs(cH_rbio37[x][y])
			sum_cV_rbio37 += abs(cV_rbio37[x][y])
			sum_cD_rbio37 += abs(cD_rbio37[x][y])
	
	avg_cH_rbio37 = sum_cH_rbio37/p*q
	avg_cV_rbio37 = sum_cV_rbio37/p*q
	avg_cD_rbio37 = sum_cD_rbio37/p*q
	
	enegy_cH_rbio37 =  pow(sum_cH_rbio37,2)/pow(p*q,2)
	enegy_cV_rbio37 =  pow(sum_cV_rbio37,2)/pow(p*q,2)
	enegy_cD_rbio37 =  pow(sum_cD_rbio37,2)/pow(p*q,2)
	
	
	# enegy =  np.sqrt(np.sum(np.array(coeffs[cD]) ** 2)) / len(coeffs[-k])

	# print("avg_cH")
	# # print(avg_cH)
	# print(avg_cH)

	# print("avg_cV")
	# # print(avg_cV)
	# print(avg_cV)

	# print("enegy")
	# # print(enegy)	
	# print(enegy)	
	
	# avg_all_cH = np.array([avg_cH_db3,avg_cH_sym3,avg_cH_rbio33,avg_cH_rbio35,avg_cH_rbio37])
	# ans_zscroce = stats.zscore(avg_all_cH)
	ans_return = np.array([avg_cH_db3,
	enegy_cV_db3,
	avg_cH_sym3,
	enegy_cV_sym3,
	avg_cH_rbio33,
	enegy_cV_rbio33,
	enegy_cD_rbio33,
	avg_cH_rbio35,
	enegy_cV_rbio35,
	enegy_cD_rbio35,
	avg_cH_rbio37,
	enegy_cH_rbio37,
	enegy_cV_rbio37,
	enegy_cD_rbio37])
	# ans_return = {
	# 'cH': avg_cH,
	# 'cV': avg_cV,
	# 'cD': avg_cD,
	# 'enegy': enegy}[ans]
	# df = (ans_return - ans_return.mean())/ans_return.std(ddof=0)
	# ans_zscroce = stats.zscore(ans_return)
	# print(ans_return)
	# print(ans_zscroce)
	# print(df)
	
	return ans_return
	
def ann(train_x, test_x, train_y, test_y,input):
	# Initialising the ANN
	classifier = Sequential()

	# Adding the input layer and the first hidden layer
	classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = input))

	# Adding the second hidden layer
	classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

	# Adding the output layer
	classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

	# Compiling the ANN
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

	# Fitting the ANN to the Training set
	classifier.fit(train_x, train_y, batch_size = 10, epochs = 1000)

	# Part 3 - Making predictions and evaluating the model

	# Predicting the Test set results
	y_pred = classifier.predict(test_x)
	y_pred = (y_pred > 0.5)

	cm = confusion_matrix(test_y, y_pred)
	print(y_pred)
	# plt.plot(cm)
	# plt.show()
	
def Aj_ODdectiect(img,fileName):
	# b,g,r = cv2.split(img)
	
	p = 0.970
	
	width, height = img.shape
	num = width*height
		
	Y = np.ravel(img)	
	Y = sorted(Y)
	
	top20 = int(round(p*len(Y)))
	threshold = Y[top20]
	
	print(threshold*100)
	
	indices = np.argwhere(img >= threshold)		
	print ("indices : ",len(indices))
	Z = np.zeros((width,height,3), np.uint8)
	# print ("Z : ",Z.shape)
	
	for i in range(len(indices)) :
		# print ("i : ",indices[i])		
		Z[indices[i][0],indices[i][1]] = (255,255,255)	
	
	
	ZC = cv2.erode(Z,np.ones((3,3),np.uint8),iterations = 1)
	ZCE = cv2.morphologyEx(ZC, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
 
	minX = np.min(indices[:,0])
	minY = np.min(indices[:,1])
	maxX = np.max(indices[:,0])
	maxY = np.max(indices[:,1])
	# print("minX",minX)
	# print("minY",minY)
	# print("maxX",maxX)
	# print("maxY",maxY)
	#print(ZCE)
	#cv2.imshow(fileName, ZCE)
	ret,th1 = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
	# cv2.imshow(fileName, th1)
	# print(th1)
	return th1
	
## import image ##
# address = "..//ICTES//Thesis//Code//od_roi//DataMining//training"
imgs , fileName = loadImagesFromFolder('G:/My Drive/ICTES/Thesis/Code/od_roi/DataMining/training/*jpg')
book = xlrd.open_workbook('G:/My Drive/ICTES/Thesis/Code/od_roi/DataMining/BiomisaData.xlsx')
sheet = book.sheet_by_name('all')

fileData = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
trainData = []
train_w_cH = []
test_w_cH = []
data = []
target = []

i = 0
for image in imgs:	
	# cv2.imshow('image',image)
	# ## split BGR layer ##
	b,g,r = cv2.split(image)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)	
	# img = img_to_matrix(r)
	# img = flatten_image(r)
	# X_normalized = preprocessing.normalize(g, norm='l1', axis=0)
	X_normalized = preprocessing.normalize(g, norm='l2')
	# io.imshow(X_normalized)
	# plt.show()
	print("X_normalized data")
	print (g)
	print (X_normalized)
	
	data.append(X_normalized)
	
	name = fileName[i].split(".")	
	name2 = name[0].split("_")
	if name2[3] == "g":
		target.append(1)
		print(name2[3])
	else: 
		target.append(0)
		print(name2[3])
	
	## send normalization image to optic detection ##		
	# trainData.append(cornerDetect(image))	
	trainData.append(Aj_ODdectiect(X_normalized,fileName[i]))	
		
	i+=1
# training, labels = getTrainingData()

# data = np.ravel(trainData)


y  = np.array(target)
# labelsMat  = np.array(fileData[1:])
data = np.array(trainData)
# print (data)
print (y)

## split into training and test part ##
# labelsMatData = labelsMat[:,1].ravel()
# is_train = np.random.uniform(0, 1, len(data)) <= 0.8
# y = np.where(np.array(labelsMatData)=="Glaucoma", 1, 0)
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size = 0.2)
print("X_train data")
print (X_train)

d2_train_dataset = flatten_image(X_train)
d2_test_dataset = flatten_image(X_test)
# print("2D data")
# print (d2_train_dataset)
# print (Y_train)
# print (Y_test)
# plt.imshow(data)
# plt.show()

# x_train, y_train = data[is_train], y[is_train]
# x_test, y_test = data[is_train==False], y[is_train==False]
# print("data")
# print (x_train)
# print (y_train)
# print (y_test)

train_y, test_y = Y_train, Y_test

## pca ##
# V,S,mean_X = pca(X_train)
# print("V")
# print(V)
# print("S")
# print(S)
# print("mean_X")
# print(mean_X)

train_x , test_x = PCAs(d2_train_dataset,d2_test_dataset,7)

## wavelet ##
for train_dataset in X_train:
	# print("train_dataset")	
	# print(train_dataset)
	# train_w_cH.append(Wavelet_avg(train_dataset,'enegy'))
	train_w_cH.append(Wavelet_avg(train_dataset))
	
for test_dataset in X_test:
	# print("test_dataset")	
	# print(test_dataset)
	# test_w_cH.append(Wavelet_avg(test_dataset,'enegy'))
	test_w_cH.append(Wavelet_avg(test_dataset))

# print("train_w_cH")
# print(train_w_cH)
train_w = stats.zscore(train_w_cH)
# train_w = Wavelet(d2_train_dataset)
test_w = stats.zscore(test_w_cH)
# test_w = Wavelet(d2_test_dataset)
# train_w = np.array(train_w)
# test_w = np.array(test_w)
# print("wavelet")
# print(test_w)

## KNN ##
print("pca")
KNN(train_x , test_x, train_y, test_y, 3)
print("wavelet")
KNN(train_w , test_w, train_y, test_y, 3)

############################
## SVM ##
print("pca")
SVM(train_x , test_x, train_y, test_y, 3)
print("wavelet")
SVM(train_w , test_w, train_y, test_y, 3)

############################
## ANN ##
# print("pca")
# ann(train_x , test_x, train_y, test_y, 5)
# print("wavelet")
# ann(train_w , test_w, train_y, test_y, 352)
cv2.waitKey(0)

