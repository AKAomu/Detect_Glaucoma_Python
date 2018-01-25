## LDA ##
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
import time

## PyWavelet ##
import pywt

## sklearn ##
from skimage import io
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif ,SelectKBest, f_regression, SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def flatten_image(data):
	"""
	takes in an (m, n) numpy array and flattens it 
	into an array of shape (1, m * n)
	"""
	
	nsamples, nx, ny = data.shape
	# nsamples = data.shape
	d2_train_dataset = data.reshape((nsamples,nx*ny))
	return d2_train_dataset

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
	
def SVM(train_x, test_x, train_y, test_y, kernels):
	# ANOVA SVM-C
	# 1) anova filter, take 3 best ranked features
	# anova_filter = SelectKBest(f_regression, k=kernels)
	# 2) svm
	clf = svm.SVC(kernel='linear')

	# anova_svm = make_pipeline(anova_filter, clf)
	# anova_svm.fit(train_x, train_y)
	clf.fit(train_x, train_y)
	# clf.fit(train_x, Y_train)
	# y_pred = anova_svm.predict(test_x)
	y_pred = clf.predict(test_x)
	target_names = ['Normal', 'Glaucoma', 'DR']
	# print("SVM")
	print(classification_report(test_y, y_pred,target_names=target_names))
	print("label test_x")
	print(test_y)
	print("predict test_x")
	print(y_pred)
	
def Wavelet_avg(data):
	## db3 ##
	sum_cH_db3 = 0	
	sum_cV_db3 = 0	
	sum_cD_db3 = 0	
	
	sum_cH_db3_energy = 0
	sum_cV_db3_energy = 0
	sum_cD_db3_energy = 0
	
	coeffs_db3 = pywt.dwt2(data, 'db3')
	cA_db3, (cH_db3, cV_db3, cD_db3) = coeffs_db3
	
	p , q = cH_db3.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_db3 += abs(cH_db3[x][y])			
			sum_cV_db3 += abs(cV_db3[x][y])			
			sum_cD_db3 += abs(cD_db3[x][y])			
			
			sum_cH_db3_energy += pow(abs(cH_db3[x][y]),2)
			sum_cV_db3_energy += pow(abs(cV_db3[x][y]),2)
			sum_cD_db3_energy += pow(abs(cD_db3[x][y]),2)
	
	avg_cH_db3 = sum_cH_db3/p*q
	avg_cV_db3 = sum_cV_db3/p*q
	avg_cD_db3 = sum_cD_db3/p*q
	
	enegy_cH_db3 =  sum_cH_db3_energy/pow(p,2) * pow(q,2)
	enegy_cV_db3 =  sum_cV_db3_energy/pow(p,2) * pow(q,2)
	enegy_cD_db3 =  sum_cD_db3_energy/pow(p,2) * pow(q,2)
	
	## sym3 ##
	sum_cH_sym3 = 0	
	sum_cV_sym3 = 0	
	sum_cD_sym3 = 0	
	
	sum_cH_sym3_energy = 0
	sum_cV_sym3_energy = 0
	sum_cD_sym3_energy = 0
	
	coeffs_sym3 = pywt.dwt2(data, 'sym3')
	cA_sym3, (cH_sym3, cV_sym3, cD_sym3) = coeffs_sym3
	
	p , q = cH_sym3.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_sym3 += abs(cH_sym3[x][y])
			sum_cV_sym3 += abs(cV_sym3[x][y])
			sum_cD_sym3 += abs(cD_sym3[x][y])
			
			sum_cH_sym3_energy += pow(abs(cH_sym3[x][y]),2)
			sum_cV_sym3_energy += pow(abs(cV_sym3[x][y]),2)
			sum_cD_sym3_energy += pow(abs(cD_sym3[x][y]),2)
	# print(sum_cH)
	# print(sum_cV)
	# print(sum_cD)
	avg_cH_sym3 = sum_cH_sym3/p*q
	avg_cV_sym3 = sum_cV_sym3/p*q
	avg_cD_sym3 = sum_cD_sym3/p*q
	
	enegy_cH_sym3 =  sum_cH_sym3_energy/pow(p,2) * pow(q,2)
	enegy_cV_sym3 =  sum_cV_sym3_energy/pow(p,2) * pow(q,2)
	enegy_cD_sym3 =  sum_cD_sym3_energy/pow(p,2) * pow(q,2)
	
	## rbio3.3 ##
	sum_cH_rbio33 = 0
	sum_cV_rbio33 = 0
	sum_cD_rbio33 = 0
	
	sum_cH_rbio33_energy = 0
	sum_cV_rbio33_energy = 0
	sum_cD_rbio33_energy = 0
	
	coeffs_rbio33 = pywt.dwt2(data, 'rbio3.3')
	cA_rbio33, (cH_rbio33, cV_rbio33, cD_rbio33) = coeffs_rbio33
	
	p , q = cH_rbio33.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_rbio33 += abs(cH_rbio33[x][y])
			sum_cV_rbio33 += abs(cV_rbio33[x][y])
			sum_cD_rbio33 += abs(cD_rbio33[x][y])
			
			sum_cH_rbio33_energy += pow(abs(cH_rbio33[x][y]),2)
			sum_cV_rbio33_energy += pow(abs(cV_rbio33[x][y]),2)
			sum_cD_rbio33_energy += pow(abs(cD_rbio33[x][y]),2)
	
	avg_cH_rbio33 = sum_cH_rbio33/p*q
	avg_cV_rbio33 = sum_cV_rbio33/p*q
	avg_cD_rbio33 = sum_cD_rbio33/p*q
	
	enegy_cH_rbio33 =  sum_cH_rbio33_energy/pow(p,2) * pow(q,2)
	enegy_cV_rbio33 =  sum_cV_rbio33_energy/pow(p,2) * pow(q,2)
	enegy_cD_rbio33 =  sum_cD_rbio33_energy/pow(p,2) * pow(q,2)
	
	## rbio3.5 ##
	sum_cH_rbio35 = 0
	sum_cV_rbio35 = 0
	sum_cD_rbio35 = 0
	
	sum_cH_rbio35_energy = 0
	sum_cV_rbio35_energy = 0
	sum_cD_rbio35_energy = 0
	
	coeffs_rbio35 = pywt.dwt2(data, 'rbio3.5')
	cA_rbio35, (cH_rbio35, cV_rbio35, cD_rbio35) = coeffs_rbio35
	
	p , q = cH_rbio35.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_rbio35 += abs(cH_rbio35[x][y])
			sum_cV_rbio35 += abs(cV_rbio35[x][y])
			sum_cD_rbio35 += abs(cD_rbio35[x][y])
			
			sum_cH_rbio35_energy += pow(abs(cH_rbio35[x][y]),2)
			sum_cV_rbio35_energy += pow(abs(cV_rbio35[x][y]),2)
			sum_cD_rbio35_energy += pow(abs(cD_rbio35[x][y]),2)
	
	avg_cH_rbio35 = sum_cH_rbio35/p*q
	avg_cV_rbio35 = sum_cV_rbio35/p*q
	avg_cD_rbio35 = sum_cD_rbio35/p*q
	
	enegy_cH_rbio35 =  sum_cH_rbio35_energy/pow(p,2) * pow(q,2)
	enegy_cV_rbio35 =  sum_cV_rbio35_energy/pow(p,2) * pow(q,2)
	enegy_cD_rbio35 =  sum_cD_rbio35_energy/pow(p,2) * pow(q,2)
	
	## rbio3.7 ##
	sum_cH_rbio37 = 0
	sum_cV_rbio37 = 0
	sum_cD_rbio37 = 0
	
	sum_cH_rbio37_energy = 0
	sum_cV_rbio37_energy = 0
	sum_cD_rbio37_energy = 0
	
	coeffs_rbio37 = pywt.dwt2(data, 'rbio3.7')
	cA_rbio37, (cH_rbio37, cV_rbio37, cD_rbio37) = coeffs_rbio35
	
	p , q = cH_rbio37.shape
	for x in range(0,p):
		for y in range(0,q):
			sum_cH_rbio37 += abs(cH_rbio37[x][y])
			sum_cV_rbio37 += abs(cV_rbio37[x][y])
			sum_cD_rbio37 += abs(cD_rbio37[x][y])
			
			sum_cH_rbio37_energy += pow(abs(cH_rbio37[x][y]),2)
			sum_cV_rbio37_energy += pow(abs(cV_rbio37[x][y]),2)
			sum_cD_rbio37_energy += pow(abs(cD_rbio37[x][y]),2)
	
	avg_cH_rbio37 = sum_cH_rbio37/p*q
	avg_cV_rbio37 = sum_cV_rbio37/p*q
	avg_cD_rbio37 = sum_cD_rbio37/p*q
	
	enegy_cH_rbio37 =  sum_cH_rbio37_energy/pow(p,2) * pow(q,2)
	enegy_cV_rbio37 =  sum_cV_rbio37_energy/pow(p,2) * pow(q,2)
	enegy_cD_rbio37 =  sum_cD_rbio37_energy/pow(p,2) * pow(q,2)
	
	
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

def calTime(startTime):
	sum = 0
	for times in startTime:
		sum+=times
	# print(sum)
	return time.process_time()-sum
	

## import image ##
imgs , fileName = loadImagesFromFolder('G:/My Drive/ICTES/Thesis/Database_Glaucoma/biomisa/all/images/*jpg')	
# Gimgs , GfileName = loadImagesFromFolder('G:/My Drive/ICTES/Thesis/Database_Glaucoma/MIAG/RIM-ONE r2/Glaucoma and glaucoma suspicious/*jpg')	
# Nimgs , NfileName = loadImagesFromFolder('G:/My Drive/ICTES/Thesis/Database_Glaucoma/MIAG/RIM-ONE r2/Normal/*jpg')	
# book = xlrd.open_workbook('G:/My Drive/ICTES/Thesis/Code/od_roi/DataMining/BiomisaData.xlsx')
# sheet = book.sheet_by_name('all')

# fileData = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
# labelsMat  = np.array(fileData[1:])
# labelsMatData = labelsMat[:,1].ravel()
# y = np.where(np.array(labelsMatData)=="Glaucoma", 1, 0)
data = []
trainData = []
target = []
test_w = []
train_w = []
times = []

target_names = ['Normal', 'Glaucoma', 'DR']
# print(y)
i = 0
j = 0
for image in imgs:	
	b,g,r = cv2.split(image)	
	X_normalized = preprocessing.normalize(g, norm='l2')	
	data.append(X_normalized)
	# print(X_normalized.shape)
	# print(fileData[j][0])
	name = fileName[j].split(".")
	# print("name "+name[0])
	# print("file "+fileData[j+1][0])
	name2 = name[0].split("_")
	# if fileData[j][0] == name[0]:
		# print("check "+name[0])
	if name2[1] == "h":
		target.append(0)
	elif name2[1] == "g": 
		target.append(1)
	elif  name2[1] == "dr":
		target.append(2)
	
		# print(name2[3])
	# data.append(Aj_ODdectiect(X_normalized,fileName[j]))	
	# if labelsMatData[j] == "Glaucoma":	
		# target.append(1) 
	# else :
		# target.append(0)
	j+=1
times.append(time.process_time())

# for image in Nimgs:
	# b,g,r = cv2.split(image)	
	# X_normalized = preprocessing.normalize(g, norm='l2')	
	# data.append(X_normalized)
	# # print(fileData[j][0])
	# # name = fileName[j].split(".")
	# # print("name "+name[0])
	# # print("file "+fileData[j+1][0])
	# # name2 = name[0].split("_")
	# # if fileData[j][0] == name[0]:
		# # print("check "+name[0])
	# # if name2[3] == "g":
	# target.append(0)	
# # print(data)
# times.append(calTime(times))

for image in data:
	# print(image.shape)
	widthBase = 250.0 / image.shape[1]
	dim = (250, int(image.shape[0] * widthBase))
	# dim = (250, 250)
	img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	# print(img.shape)
	trainData.append(img)
times.append(calTime(times))

# print(trainData)	
data = np.array(trainData)
target = np.array(target)
print(data.shape)
print(target)
# X = flatten_image(data)
# X = data

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size = 0.2)

d2_train_dataset = flatten_image(X_train)
d2_test_dataset = flatten_image(X_test)

times.append(calTime(times))

### PCA ###
pca = PCA(n_components=6)
X_r = pca.fit_transform(d2_train_dataset)
print(X_r)

pca2 = PCA(n_components=6)
X_r2 = pca.transform(d2_test_dataset)

print("PCA+SVM")
SVM(X_r , X_r2, Y_train, Y_test, 3)

times.append(calTime(times))
print("Process time of PCA+SVM : "+str(times[3]))

### LDA ###
lda3 = LinearDiscriminantAnalysis(n_components=6)
X_r3 = lda3.fit(d2_train_dataset, Y_train).transform(d2_train_dataset)

lda4 = LinearDiscriminantAnalysis(n_components=6)
X_r4 = lda4.fit(d2_test_dataset, Y_test).transform(d2_test_dataset)

print("LDA+SVM")
SVM(X_r3 , X_r4, Y_train, Y_test, 1)

times.append(calTime(times))
print("Process time of LDA+SVM : "+str(times[4]))

### PCA+LDA ###

lda5 = LinearDiscriminantAnalysis(n_components=6)
X_r5 = lda5.fit(X_r, Y_train).transform(X_r)

lda6 = LinearDiscriminantAnalysis(n_components=6)
X_r6 = lda5.fit(X_r2, Y_test).transform(X_r2)

print("PCA+LDA+SVM")
# print(classification_report(Y_test, lda.predict(X_r2),target_names=target_names))
# print(Y_test)
# print(lda.predict(X_r2))
SVM(X_r5 , X_r6, Y_train, Y_test, 1)

# print("SVM")
# print(classification_report(X_r6, Y_test,target_names=target_names))
# print("label test_x")
# print(X_r6)
# print("predict test_x")
# print(Y_test)

times.append(calTime(times))
print("Process time of PCA+LDA+SVM : "+str(times[5]))
# print(X_r5.shape)

### wavelet + PCA ###

X_r_w = X_r
X_r2_w = X_r2
ans_w_train = []
ans_w_test = []
# print(X_r_w)
a = 0
b = 0
for train_dataset in X_train:
	# print("train_dataset")	
	# print(train_dataset)
	# train_w.append(Wavelet_avg(train_dataset))
	ans_w_train.append(np.append(X_r_w[a], Wavelet_avg(train_dataset)))
	# ans_w_train[a].append(Wavelet_avg(train_dataset))
	a+=1
	
	
for test_dataset in X_test:
	# print("test_dataset")	
	# print(test_dataset)
	# test_w.append(Wavelet_avg(test_dataset))
	ans_w_test.append(np.append(X_r2_w[b],Wavelet_avg(test_dataset)))
	b+=1
	


train_w_cH = np.array(ans_w_train)
test_w_cH = np.array(ans_w_test)

# train_w_cH = stats.zscore(train_w_cH)
# test_w_cH = stats.zscore(test_w_cH)

# print(test_w_cH)
# print(Y_train)
# plt.plot(train_w_cH[:,0],'b',train_w_cH[:,1],'r')
# pl.legend()
# pl.show()

### LDA Classification ###
# lda = LinearDiscriminantAnalysis()
# lda.fit(X_r,Y_train)

# print(X_r6)

### Classification ###


print("wavelet+PCA+SVM")
# print(train_w_cH.shape)
SVM(train_w_cH , test_w_cH, Y_train, Y_test, 1)



# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

times.append(calTime(times))
print("Process time of wavelet+SVM : "+str(times[6]))
print(times)
