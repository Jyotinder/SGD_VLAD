#!/usr/local/bin/python2.7
import cv2
import os
from Vlad import *
from scipy.cluster.vq import *
from sklearn.decomposition import PCA

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def siftPyramid(image_path):

    """
        Input:  image_path
                path to raw image (./image/airplane/airplane1.tif)
        Output: fd
                SIFT FV
        Note:: To configure HOG parameter use config file to set
                orientations, pixels_per_cell, cells_per_block, visualize, normalize
                size of fd depend upoin these paprameter
    """
    print "##################"
    print "SIFT Enter"+image_path
    im = cv2.imread(image_path)
    rows,cols,ch = im.shape
    if rows!= 256 or cols !=256:
        print image_path
        print rows, cols
        im = cv2.resize(im,(256, 256), interpolation = cv2.INTER_CUBIC)
        return []
    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    split= blockshaped(im, 256/4, 256/4)
    descriptors = np.array([], dtype=np.float).reshape(0,128)
    #fd=[]
    for image in split:
        kpts = fea_det.detect(image)
        kpts, des = des_ext.compute(image, kpts)
        if des !=[] and des is not None :
            descriptors=np.vstack([descriptors,des])
            #fd.append(des)
    return descriptors

def sift(image_path):
    """
        Input:  image_path
                path to raw image (./image/airplane/airplane1.tif)
        Output: fd
                SIFT FV
        Note:: To configure HOG parameter use config file to set
                orientations, pixels_per_cell, cells_per_block, visualize, normalize
                size of fd depend upoin these paprameter
    """
    print "##################"
    print "SIFT Enter"+image_path
    im = cv2.imread(image_path)
    rows,cols,ch = im.shape
    if rows!= 256 or cols !=256:
        print image_path
        print rows, cols
        im = cv2.resize(im,(256, 256), interpolation = cv2.INTER_CUBIC)
        return []
    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    print "##################"
    print "SIFT End"
    return des

def PCA_image(image_path):
    im = cv2.imread(image_path)
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    k=128
    print "PCA Enter"+image_path
    pca = PCA(n_components=k)
    # X is the matrix transposed (n samples on the rows, m features on the columns)
    pca.fit(gray_image)
    print "PCA End"

    return pca.transform(gray_image)

def shiftToVlad(des):
    k=128
    voc, variance = kmeans(des, k, 1)
    vladVector= vladFun(des,voc)
    if k*128!=vladVector.size:
        print vladVector.size
        return None
    return vladVector








