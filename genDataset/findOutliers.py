################## findOutliers.py #########################
# Simple functions that cast out depth outliers.
#
#

import OpenEXR
from PIL import Image as Image
import Imath
import numpy as np
import array
import cv2 as cv

##################
# Reads EXR file #
##################
def openexr(filename):
    '''
    fileInfo = OpenEXR.InputFile(filename)
    dw = fileInfo.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depthStr = fileInfo.channel('R', FLOAT)
    depth = Image.fromstring(depthStr, dtype = np.float32)
    depth.shape = (sz[1], sz[0])
    '''
    depth = cv.imread(filename, cv.IMREAD_ANYDEPTH)

    return depth

########################################
# Calculates percentage over threshold #
########################################
def percentageOverThreshold(depthMap, threshold):
    #count = np.where(depthMap > threshold).sum()
    count = len(depthMap[ depthMap > threshold])
    elementCount = depthMap.shape[0] * depthMap.shape[1]
    #elementCount = len(depthMap)
    return count / elementCount * 100

#########################################
# Calculates percentage under threshold #
#########################################
def percentageUnderThreshold(depthMap, threshold):
    #count = len(np.where(depthMap < threshold))
    count = len(depthMap[ depthMap < threshold])
    elementCount = depthMap.shape[0] * depthMap.shape[1]
    #elementCount = len(depthMap)
    return count / elementCount * 100
