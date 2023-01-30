import radiomics
import cv2 as cv
import numpy as np
import joblib


features_to_extract = ['blue_iqr', 'original_firstorder_Entropy',
       'original_firstorder_Skewness',
       'original_gldm_LargeDependenceLowGrayLevelEmphasis',
       'original_glszm_ZoneEntropy', 'original_ngtdm_Complexity',
       'original_ngtdm_Strength']



def extract_features(image):
    features = [0,0,0,0,0,0,0]
    im = cv.imread('./images/' + image)
    im_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    blue_channel = im[:,:,2].flatten()[im[:,:,2].flatten().nonzero()]
    median = np.median(blue_channel) # get median
    q3, q1 = np.percentile(blue_channel, [75,25]) # get 3rd quartile and 1st quartile
    blue_iqr = q3 - q1 # calculate quartile diference
    features[0] = blue_iqr

    #Entropy
    eps = np.spacing(1)
    entropy = -1.0 * np.sum(im_gray * np.log2(im_gray + eps), 1)
    entropy = np.sum(entropy)
    features[1] = entropy


    

    print(features)
    return features


def diagnose(features):
    filename = 'svcmelanoma'
    model = joblib.load(filename)

    dia = model.predict_proba([features])
    print(dia)
    return dia





