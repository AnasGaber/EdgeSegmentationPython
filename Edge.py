
import cv2
import numpy
import sys
from matplotlib import pyplot as plt

#color correction using Grey World algorithms (2) in the assignment pdf
def gwa(orgimg):
    orgimg = orgimg.transpose(2, 0, 1).astype(numpy.uint32)
    mu_g = numpy.average(orgimg[1])
    orgimg[0] = numpy.minimum(orgimg[0]*(mu_g/numpy.average(orgimg[0])),255)
    orgimg[2] = numpy.minimum(orgimg[2]*(mu_g/numpy.average(orgimg[2])),255)
    finimg = orgimg.transpose(1, 2, 0).astype(numpy.uint8)
    #remove "#" to review the image after Grey World and before the masking
    #cv2.imshow('Grey World',finimg)
    return  finimg
#image analysis i converted the image into 3 different channals R,G,B in greyscale (3) in the assignment pdf
def img_ana(img):
    b, g, r = cv2.split(img)
#remove "#" to review the image in R,G,B channals before the masking
    #cv2.imshow("B", b)
    #cv2.imshow("G", g)
    #cv2.imshow("R", r)
#this is used to represent the values of each channel in graph
    plt.hist(b.ravel(), 256, [0, 256])
    plt.hist(g.ravel(), 256, [0, 256])
    plt.hist(r.ravel(), 256, [0, 256])
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
#remove "#" to review the graph of R,G,B channals
    #plt.show()

#thresholding function that take the image, min values and max values of the skin in each image in RGB & YCbCr (4,5) in the assignment pdf
def img_mask(orgimg):
    #convert the image to HSV
    mimg=cv2.cvtColor(orgimg, cv2.COLOR_BGR2HSV)
    #convert the image to YCbCr
    yimg=cv2.cvtColor(orgimg, cv2.COLOR_BGR2YCR_CB)
    #masking using HSV/RGB
    minBGR = numpy.array([0, 30, 0], dtype = "uint8")
    maxBGR = numpy.array([50, 255, 255], dtype = "uint8")
    maskBGR = cv2.inRange(mimg,minBGR,maxBGR)
    #displying the masked version of the image in black and white
    cv2.imshow('maskRGB',maskBGR)
    #displyin the masked version with the skin color
    skinBGR = cv2.bitwise_and(mimg, mimg, mask = maskBGR)
    cv2.imshow('skinRGB',cv2.cvtColor(skinBGR, cv2.COLOR_HSV2BGR))
    #masking using YCbCr
    minYCbCr = numpy.array((0, 130, 85), dtype = "uint8")
    maxYCbCr = numpy.array([255,173,127], dtype = "uint8")
    maskYCbCr = cv2.inRange(yimg,minYCbCr,maxYCbCr)
    #displying the masked version of the image in black and white
    cv2.imshow('maskYCbCr',maskYCbCr)
    skinYCbCr = cv2.bitwise_and(yimg, yimg, mask = maskYCbCr)
    #displyin the masked version with the skin color
    cv2.imshow('skinYCbCr',cv2.cvtColor(skinYCbCr, cv2.COLOR_YCR_CB2BGR))
#reading the images (1) in the assignment pdf
#1st image
img0 = cv2.imread('0_rendered.png', -1)
img01= gwa(img0)
img_ana(img01)
img_mask(img01)
cv2.waitKey(5000) #disply image for 5 sec
#2nd image
img1 = cv2.imread('1_rendered.png', -1)
img11= gwa(img1)
img_ana(img11)
img_mask(img11)
cv2.waitKey(5000) #disply image for 5 sec
#3rd image
img2 = cv2.imread('3_rendered.png', -1)
img21= gwa(img2)
img_ana(img21)
img_mask(img21)
cv2.waitKey(5000) #disply image for 5 sec
#4th image
img3 = cv2.imread('4_rendered.png', -1)
img31= gwa(img3)
img_ana(img31)
img_mask(img31)
cv2.waitKey(5000) #disply image for 5 sec
#5th image
img4 = cv2.imread('5_rendered.png', -1)
img41= gwa(img4)
img_ana(img41)
img_mask(img41)
cv2.waitKey(5000) #disply image for 5 sec
#6th image
img5 = cv2.imread('6_rendered.png', -1)
img51= gwa(img5)
img_ana(img51)
img_mask(img51)
cv2.waitKey(5000) #disply image for 5 sec
#7th image
img6 = cv2.imread('7_rendered.png', -1)
img61= gwa(img6)
img_ana(img61)
img_mask(img61)
cv2.waitKey(5000) #disply image for 5 sec
#8th image
img7 = cv2.imread('8_rendered.png', -1)
img71= gwa(img7)
img_ana(img71)
img_mask(img71)
cv2.waitKey(5000) #disply image for 5 sec
#9th image
img8 = cv2.imread('9_rendered.png', -1)
img81= gwa(img8)
img_ana(img81)
img_mask(img81)
cv2.waitKey(5000) #disply image for 5 sec
#10th image
img9 = cv2.imread('10_rendered.png', -1)
img91= gwa(img9)
img_ana(img91)
img_mask(img91)
cv2.waitKey(5000) #disply image for 5 sec

cv2.waitKey(0)
cv2.destroyAllWindows()
