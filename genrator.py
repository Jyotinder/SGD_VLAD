from sklearn.cross_validation import train_test_split
import cv2
import os
import numpy as np

from sklearn.svm import LinearSVC
#from scipy.cluster.vq import *
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from xlrd.compdoc import x_dump_line

from Sift import *
import itertools
import random
from sklearn.cross_validation import KFold



def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]

def vocVlad(des):
    k = 32
    k_means = KMeans(init='k-means++', n_clusters=k, n_init=1)
    k_means.fit(des)
    voc = k_means.cluster_centers_
    return voc

def trainTestSet(setX,setY):
    x_trainingList=[]
    y_train=[]
    descriptors = np.array([], dtype=np.float).reshape(0,128)
    des_list = []
    for i,image_path in enumerate(setX):
        # reduce the column of the image to 128 doesn't change the number of rows
        des=sift(image_path)
        try:
            if des !=[] and des is not None:
                descriptors=np.vstack([descriptors,des])
                des_list.append((image_path,setY[i],des))
            else:
                    del setX[i]
                    del setY[i]
        except:
            print "Exception for "+ image_path
            del setX[i]
            del setY[i]
    print "Going for VOC"
    voc = vocVlad(descriptors)
    print "END for VOC"
    for i in xrange(len(des_list)):
        print "calculating vlad for"+ des_list[i][0]
        x_trainingList.append(vladFun(des_list[i][2],voc))
        y_train.append(des_list[i][1])
    return x_trainingList,y_train
def getrows(row,X_train,Y_train):
    X=[]
    Y=[]
    for i in row:
        if i<len(X_train):
            X.append(X_train[i])
            Y.append(Y_train[i])
    return X,Y

def iter_minibatches(chunksize,X_train,Y_train):
    # Provide chunks one by one
    chunkstartmarker = 0
    print len(X_train)
    while chunkstartmarker < len(X_train):
        chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        X_chunk, y_chunk = getrows(chunkrows,X_train,Y_train)
        yield X_chunk, y_chunk
        chunkstartmarker += chunksize

def direcrtoryProcessing(train_path):
    training_names = os.listdir(train_path)
    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:#Got three directory having images
        dir = os.path.join(train_path, training_name)
        class_path = imlist(dir)
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path)
        class_id+=1
    X,Y=trainTestSet(image_paths,image_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
    clf = SGDClassifier()
    batcherator = iter_minibatches(10,X_train,y_train)
    for X_chunk, y_chunk in batcherator:
        clf.partial_fit(X_chunk, y_chunk,classes=np.unique(Y))
        y_predicted = clf.predict(X_test)
        print(classification_report(y_test,y_predicted,target_names=training_names ))





def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, Cname=""):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(Cname))
    plt.xticks(tick_marks, Cname, rotation=45)
    plt.yticks(tick_marks, Cname)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':
    direcrtoryProcessing("./Images")