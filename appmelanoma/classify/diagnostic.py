from radiomics import featureextractor

import cv2 as cv
import numpy as np
import joblib
from scipy.stats import skew
import os
import SimpleITK as sitk
from iteration_utilities import flatten, deepflatten

features_to_extract = ['blue_iqr', 'original_firstorder_Entropy',
       'original_firstorder_Skewness',
       'original_gldm_LargeDependenceLowGrayLevelEmphasis',
       'original_glszm_ZoneEntropy', 'original_ngtdm_Complexity',
       'original_ngtdm_Strength']



def extract_features(image):
    features = []
    im = cv.imread('./images/' + image)
    im_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    mask_path = './app_test_mask/'
    mask = cv.imread(mask_path + image[:-4] + '_segmentation.png')
    extractor = featureextractor.RadiomicsFeatureExtractor()# create extractor instance from Pyradiomics
    im, label = './pyradiomicsdir/img_gray_1.jpg' , './pyradiomicsdir/mask_1.jpg'

    # save grayscale image and mask as jpg
    cv.imwrite(im, im_gray) 
    cv.imwrite(label, mask)

    result = extractor.execute(im, label) # extract features and save in OrdDict result
    values = list(result.values()) # save all values in a list. These values are stored in np.arrays and must be converted to floats 
    keys = list(result.keys()) # save result keys in a list. These values are stored in np.arrays and must be converted to floats
    v = [] # array to store values
    k = [] # array to store keys (these will be the column names in the dataframe)

    blue_channel = im_rgb[:,:,2].flatten()[im_rgb[:,:,2].flatten().nonzero()]
    median = np.median(blue_channel) # get median
    q3, q1 = np.percentile(blue_channel, [75,25]) # get 3rd quartile and 1st quartile
    blue_iqr = q3 - q1 # calculate quartile diference
    features.append(blue_iqr)

    # iterate through values and append them to list
    for i in deepflatten(values[22:]):
        v.append(i.tolist())
    # iterate through the keys and append them to list
    for l in keys[22:]:   
        k.append(l)

    for c, k in enumerate(k):
        if k in features_to_extract:
            features.append(v[c])

    return features





def diagnose(features):
    filename = 'svcmelanoma'
    model = joblib.load(filename)

    dia = model.predict_proba([features])
    return dia





