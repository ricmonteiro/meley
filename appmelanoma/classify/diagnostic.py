from radiomics import featureextractor

import cv2 as cv
import numpy as np
import joblib
from scipy.stats import skew
import os
import SimpleITK as sitk
'''
features_to_extract = ['blue_iqr', 'original_firstorder_Entropy',
       'original_firstorder_Skewness',
       'original_gldm_LargeDependenceLowGrayLevelEmphasis',
       'original_glszm_ZoneEntropy', 'original_ngtdm_Complexity',
       'original_ngtdm_Strength']
'''


def extract_features(image):
    print(os.getcwd())
    features = [0,0,0,0,0,0,0]
    im = cv.imread('./images/' + image)
    im_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    mask_path = './ISIC-2017_Training_Part1_GroundTruth/'
    print(mask_path + image[:-4]+ '_segmentation.png')
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
    features[0] = blue_iqr




    return features


'''
    #Entropy
    eps = np.spacing(1)
    entropy = -1.0 * np.sum(im_gray * np.log2(im_gray + eps), 1)
    entropy = np.sum(entropy)
    features[1] = entropy

    #Skewness
    skewness = skew(im_gray.flatten(), axis=0, bias=True)
    print(skewness)

    features[2] = skewness

    #LargeDependenceLowGrayLevelEmphasis


    #ZoneEntropy


    #Complexity


    #Strength'''  



def diagnose(features):
    filename = 'svcmelanoma'
    model = joblib.load(filename)

    dia = model.predict_proba([features])
    return dia





