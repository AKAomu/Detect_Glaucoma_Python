## import ##
## openCV ##
import cv2 
import numpy as np
import os
import glob
import ntpath
from matplotlib import pyplot as plt
## Scikit-image ##
# from skimage.feature import local_binary_pattern
# from scipy.stats import itemfreq
# from sklearn.preprocessing import normalize
# import cvutils
# from numpy._distributor_init import NUMPY_MKL
# from skimage.measure import structural_similarity as ssim
# ## GPU ##
#import pycuda.driver as cuda
#import pycuda.autoinit
#from pycuda.compiler import sourcemodule

def clearBloodVessels(img1,img2):
	## split BGR layer ##
	b,g,r = cv2.split(img1)
	b_m,g_m,r_m = cv2.split(img2)
	
	g_new = g_m # set g_new = g layer of blood vessels
	# g_new = 1 - g_m
	# (thresh, im_bw) = cv2.threshold(g_m, 255, 128, cv2.THRESH_BINARY | cv2.THRESH_OTSU)	
	
	# # new_img = g * im_bw
	# new_img = cv2.multiply(g,g_new)
	
	# blur_g = cv2.medianBlur(new_img,7)
		
	# cv2.imshow('g_new',g_new)
	# cv2.imshow('result',blur_g)
	# print "new_img : ", new_img
	# print "median : ", blur_g
	
	new_img = g # set new_img = g layer of original image
	## find and check if g_new equal white set new_img equal white ##
	for i in range(len(g)):
		for j in range(len(g[i])):					
			if g_new[i][j] == 255 :			
				new_img[i][j] = 255		
		
	## blur image with meadian blur ##
	blur_g = cv2.medianBlur(new_img,7)	
	
	## display ##
	# cv2.imshow('new_img',new_img)
	# cv2.imshow('result',blur_g)
	print ("new_img : ", new_img)
	print ("median : ", blur_g)
	# cv2.imwrite("blur_g.jpg",blur_g)
	
	cv2.waitKey(0)
	return blur_g
	
def opticDetection(imgOriginal,imgNormalization):
	## convert to grayscale ##
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	## split BGR layer ##
	# b,g,r = cv2.split(img)
	
	## threshold to get just the signature ##
	#retval, thresh_gray = cv2.threshold(imgNormalization, thresh=253, maxval=255, type=cv2.THRESH_BINARY)
	retval, thresh_gray = cv2.threshold(imgNormalization, thresh=177, maxval=255, type=cv2.THRESH_BINARY)
	# crop = thresh_gray
	
	## find where the signature is and make a cropped region ##
	idx = (thresh_gray == 255)
	# idx = (imgNormalization >= 177)&(imgNormalization < 180)
	points = np.argwhere(idx) # find where the black pixels are
	points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
	x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
	x, y, w, h = x-30, y-30, w+45, h+45 # make the box a little bigger
	# crop = imgNormalization[y:y+h, x:x+w] # create a cropped region of the image
	crop = imgNormalization[y:y+h, 0:700] # create a cropped region of the image	
	small_crop = cv2.resize(crop, (700,65), interpolation = cv2.INTER_AREA)

	# ## get the thresholded crop ##
	# # retval, thresh_crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

	# ## display ##
	# # cv2.imshow("Cropped and thresholded image", crop) 
	# # cv2.waitKey(0)
	return small_crop
		
# def addWeightedImg(img1,img2):
	# ## add weight small_bg and small_mask ##
	# mask_bg  = cv2.addWeighted(small_bg,0.8,small_mask,0.2,0)
	# # cv2.imshow('mask_bg',mask_bg)

	# img_mask = cv2.addWeighted(small,0.8,small_mask,0.2,0)

	# # edges = cv2.Canny(g,50,150)
	# # cv2.imshow('edges',edges)

	# ## add weight BGR layer and median filter ##
	# img_mask_g = cv2.addWeighted(g,0.8,g_m,0.2,0)
	# # img_mask_g = g-g_m
	# blur_g = cv2.medianBlur(img_mask_g,9)
	# img_mask_b = cv2.addWeighted(b,0.8,b_m,0.2,0)
	# # img_mask_b = b-b_m
	# blur_b = cv2.medianBlur(img_mask_b,9)
	# img_mask_r = cv2.addWeighted(r,0.8,r_m,0.2,0)
	# # img_mask_r = r-r_m
	# blur_r = cv2.medianBlur(img_mask_r,9)

	# # ##histogram equalization by openCV###
	# # equ_r = cv2.equalizeHist(blur_r)

	# img_mask_r_r = cv2.addWeighted(blur_r,0.7,img_mask_r,0.3,0)
	# img_mask_r_b = cv2.merge((blur_b,blur_g,blur_r))

	# cv2.imshow('result_1',blur_g)
	# # cv2.imshow('img_mask_g',img_mask_g)
	# # cv2.imshow('img_mask_r',img_mask_r)
	# # cv2.imshow('img_mask_b',img_mask_b)

	# ## compute the average intensity of an image ##
	# avg = np.average(small)
	# print "avg : ",avg/255.0

	# cv2.waitKey(0)
	
def avgIntensityMeanAndSD(img,num):
	## compute the average intensity of an image ##
	avg = np.average(img)/255.0
	
	## compute color mean and standard deviation  ##
	(means, stds) = cv2.meanStdDev(img)
	
	print ("Result of ", num , "\navg : ", avg , "\nmean : " , means ,"\nsd : " , stds)
	
def loadImagesFromFolder(folder):
	images = []
	fileName = []
	for filename in glob.glob(folder):
		img = cv2.imread(filename)
		if img is not None:
			images.append(img)
			fileName.append(ntpath.basename(filename))
	return images , fileName
	
def normalizationImage(img) :
	## normalization image ##	
	cv2.normalize(img , img, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)	
	return img

def histogramEqu(img):
	return cv2.equalizeHist(img)

def claheEqu(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cal = clahe.apply(img)
	return cal

def calHistogram(img):
	return cv2.calcHist([img],[1],None,[256],[0,256])

def maxFilter(img):
	# split the image into its BGR components
	b,g,r = cv2.split(img)
 
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	m = np.maximum(np.maximum(r, g), b)
	r[r < m] = 0
	g[g < m] = 0
	b[b < m] = 0
 
	# merge the channels back together and return the image
	mergeImg = cv2.merge([b, g, r])
	return mergeImg
	
def imagePyramid(img) :	
	# tmp = img
	# dst = tmp
	# cv2.pyrDown( tmp, dst)
	lower_reso = cv2.pyrDown(img)
	# generate Gaussian pyramid for A
    # G = img.copy()
    # gpA = [G]
	# for i in xrange(6):
		# G = cv2.pyrDown(G)
		# gpA.append(G)
	return lower_reso
	
# def LBP(img):
	# # X_test = []
	# # X_name = []
	# # y_test = []
	# radius = 3
    # # Number of points to be considered as neighbourers 
	# no_points = 8 * radius
    # # Uniform LBP is used
	# lbp = local_binary_pattern(img, no_points, radius, method='uniform')
	# x = itemfreq(lbp.ravel())
    # # Normalize the histogram
	# hist = x[:, 1]/sum(x[:, 1])
	# # Append image path in X_name
	# # X_name.append(img)
    # # Append histogram to X_name
	# X_test = hist
    # # Append class label in y_test
	# # y_test.append(train_dic[os.path.split(img)[1]])
	# return lbp
	
def houghTransform(img) :
	## Hough transform ##			
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
	# circles = np.uint16(np.around(circles))	
	return circles
	
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title, num):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# # setup the figure
	# fig = plt.figure(title)
	# plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# # show first image
	# ax = fig.add_subplot(1, 2, 1)
	# plt.imshow(imageA, cmap = plt.cm.gray)
	# plt.axis("off")

	# # show the second image
	# ax = fig.add_subplot(1, 2, 2)
	# plt.imshow(imageB, cmap = plt.cm.gray)
	# plt.axis("off")

	# # show the images
	# plt.show()
	print ("\nMSE : ", m , "\nSSIM : " , s) 
	# print "\nSSIM : " , s ,"\n"
	
def ODDectection(img):
	blur = cv2.blur(img,(31,31))
	return blur

def MaxMin(img,fileName):
	out_path  = 'od_roi/check/'
	## convert it to grayscale ##
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.normalize(gray,gray,0,255,cv2.NORM_MINMAX)
	# the area of the image with the largest intensity value
	blur = cv2.GaussianBlur(gray, (31,31), 0)
	(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)
	## bright ##
	cv2.circle(img, maxLoc, 41, (255, 0, 0), 2)
	## drak ##
	print ("maxVal : ",maxLoc,"\nminVal : ",minLoc)
	# cv2.circle(img, minLoc, 41, (255, 0, 0), 2)
	
	## crop image##	
	w, h = maxLoc # create a rectangle around those points
	x, y, w, z = w-45, h-45, w+45, h+60 # make the box a little bigger
	# crop = img[y:z, 0:700] # create a cropped region of the image	
 
	# display the results of the naive attempt
	# cv2.imshow("Naive", img)
	name = fileName.split(".")
	cv2.imwrite(out_path+name[0]+"_full.jpg",img)
	# return crop
	
def Aj_ODdectiect(img,fileName):
	out_path  = 'od_roi/DataMining/'
	b,g,r = cv2.split(img)
	
	p = 0.995
	
	width, height = g.shape
	num = width*height
		
	Y = np.ravel(g)	
	Y = sorted(Y)
	
	top20 = int(round(p*len(Y)))
	threshold = Y[top20]
	
	indices = np.argwhere(g >= threshold)		
	print ("indices : ",len(indices))
	Z = np.zeros((width,height,3), np.uint8)
	# print ("Z : ",Z.shape)
	
	for i in range(len(indices)) :
		# print ("i : ",indices[i])		
		Z[indices[i][0],indices[i][1]] = (255,255,255)	
	
	
	ZC = cv2.erode(Z,np.ones((3,3),np.uint8),iterations = 1)
	ZCE = cv2.morphologyEx(ZC, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
	# cv2.imshow(fileName, ZCE)
	minX = np.min(indices[:,0])
	minY = np.min(indices[:,1])
	maxX = np.max(indices[:,0])
	maxY = np.max(indices[:,1])
	print("minX",minX)
	print("minY",minY)
	print("maxX",maxX)
	print("maxY",maxY)
	
	x, y, w, z = minX, minY, maxX, maxY # make the box a little bigger
	crop = img[x:w, 0:700] # create a cropped region of the image
	# cv2.imshow(fileName, crop)
	name = fileName.split(".")
	cv2.imwrite(out_path+"python_RangeTH_"+name[0]+".jpg",crop)

	# return ZCE
	
	
## import image ##
imgs , fileName = loadImagesFromFolder('G:/My Drive/ICTES/Thesis/Database_Glaucoma/biomisa/all/images/*jpg')
# opticImg = []
i = 0
for img in imgs:
	# if "_h.jpg" in fileName[i]:
	widthBase = 700.0 / img.shape[1]
	dim = (700, int(img.shape[0] * widthBase))
	small = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		# small2 = small
		# maxFilterImg = maxFilter(small)	
		
		# ## split BGR layer ##
		# b,g,r = cv2.split(small)
		# b_m,g_m,r_m = cv2.split(maxFilterImg)			
		
		# # cv2.imshow(fileName[i],small)
		# grayImg = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) 		
		# ## Equalize the Histogram ##
		# equImg = histogramEqu(grayImg)
		
		## CLAHE Function##
		# claheImg = claheEqu(g)
		
		# ## Hough transform ##
		# blurImg = cv2.medianBlur(equImg, 25)
		# cimg = cv2.cvtColor(blurImg,cv2.COLOR_GRAY2BGR)	

		# circles = houghTransform(blurImg)
		# for j in circles[0,:]:
			# # draw the outer circle
			# cv2.circle(cimg,(j[0],j[1]),j[2],(0,255,0),2)
			# # draw the center of the circle
			# cv2.circle(cimg,(j[0],j[1]),2,(0,0,255),3)
		
		# cv2.imwrite(fileName[i],claheImg)		
		# cv2.imshow(fileName[i],cimg)
		
		# ## Pyramig Function (Down Sampling) and LBP ## 
		# pyr = imagePyramid(equImg)		
		# lbp = LBP(pyr)
		# cv2.imshow(fileName[i],lbp)
		
		## send normalization image to optic detection ##	
		# Aj_ODdectiect(small,fileName[i])
	MaxMin(small,fileName[i])
					
		# opticImg.append(Aj_ODdectiect(small,fileName[i]))
		# opticImg.append(ODDectection(g))
		# opticImg.append(opticDetection(g,claheImg))
		# opticDetection(equ)
		
		# ## calculate Histogram ##
		# grayImg = cv2.cvtColor(opticImg, cv2.COLOR_BGR2GRAY)
		# hist = calHistogram(grayImg)
		# cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
		# plt.plot(hist)
		# plt.xlim([0,256])	
		
	i+=1
plt.show()
del imgs , fileName  ,i

cv2.waitKey(0)

# ## show image ##
# j = 0
# for imgOptic in opticImg:
	# j+=1		
	# cv2.imshow(str(j),imgOptic)
	
	# ## compare the images and compute avg ##
	# avgIntensityMeanAndSD(imgOptic,j)
	# # compare_images(opticImg[0], imgOptic, "Original vs. Other",j)
	
    # ## calculate Histogram ##
	# hist = calHistogram(imgOptic)
	# plt.plot(hist)
	# plt.xlim([0,256])
		

# plt.show()	
# del imgOptic
# cv2.waitKey(0)


# #path = 'c:/users/aomu/google drive/ictes/thesis/database_glaucoma/stare/*ppm'
# #for filename in glob.glob(os.path.join(path)):
# #    img = open(filename, "r", encoding="utf-8")
# #    fileName.append(ntpath.basename(filename))
# imgs , filename = loadImagesFromFolder('C:/Users/Aomu/Google Drive/ICTES/Thesis/Database_Glaucoma/STARE/img/*ppm')
# opticImg2 = []
# for i in range(1,20):
	# img = imgs[i]
	# widthBase = 700.0 / img.shape[1]
	# dim = (700, int(img.shape[0] * widthBase))
	# small = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	# #cv2.imshow(filename[i],img)
    # ## split BGR layer ##
	# # b,g,r = cv2.split(small)

    # ## CLAHE Function##
	# # claheImg = claheEqu(g)

	# opticImg2.append(MaxMin(small))
	# # opticImg2.append(opticDetection(g,claheImg))
    # ## calculate Histogram ##
	# hist = calHistogram(small)
	# plt.plot(hist)
	# plt.xlim([0,256])
# plt.show()
# del imgs , filename ,small ,i

# ## show image ##
# j = 0
# for imgOptic in opticImg2:
	# j+=1		
	# cv2.imshow(str(j),imgOptic)
	
	# ## compare the images and compute avg ##
	# avgIntensityMeanAndSD(imgOptic,j)
	# # compare_images(opticImg2[0], imgOptic, "Original vs. Other",j)
	
    # ## calculate Histogram ##
	# hist = calHistogram(imgOptic)
	# plt.plot(hist)
	# plt.xlim([0,256])	

# plt.show()	
# del imgOptic
# cv2.waitKey(0)


# small = cv2.resize(img, (0,0), fx=0.2, fy=0.2) 
# small_mask = cv2.resize(mask, (0,0), fx=0.2, fy=0.2) 
# # small_bg = cv2.resize(bg, (0,0), fx=0.2, fy=0.2) 
# print "img size : ", small.shape
# # print "img : ", small
# # print "mask : ", small_mask

# ## use feature fuction ##
# # new_img = clearBloodVessels(small,small_mask)
# opticDetection(small)
# # avgIntensityMeanAndSD(new_img)
