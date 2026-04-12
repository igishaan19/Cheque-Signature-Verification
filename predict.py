import numpy as np
import cv2
from PIL import Image
from scipy.cluster.vq import vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
import imagehash
import os

import preproc
import features

def extract_features(image_path):
    """Extract geometric and SIFT features from an image."""
    try:
        prep_img = preproc.preproc(image_path, display=False)
        phash = int(str(imagehash.phash(Image.open(image_path))), 16)
        
        aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
            features.get_contour_features(prep_img.copy(), display=False)
            
        ratio = features.Ratio(prep_img.copy())
        centroid_0, centroid_1 = features.Centroid(prep_img.copy())
        eccentricity, solidity = features.EccentricitySolidity(prep_img.copy())
        (skewness_0, skewness_1), (kurtosis_0, kurtosis_1) = features.SkewKurtosis(prep_img.copy())
        
        # Avoid division by zero
        if bounding_rect_area == 0:
            bounding_rect_area = 1.0

        geom_features = [
            aspect_ratio, 
            convex_hull_area/bounding_rect_area,
            contours_area/bounding_rect_area,
            ratio, centroid_0, centroid_1, eccentricity, solidity,
            skewness_0, skewness_1, kurtosis_0, kurtosis_1
        ]
        
        try:
            sift_detector = cv2.xfeatures2d.SIFT_create()
        except AttributeError:
            sift_detector = cv2.SIFT_create()
            
        _, des = sift_detector.detectAndCompute(prep_img, None)
        return geom_features, des, True
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None, False

def load_model_components():
    """Load the pre-trained model, vocabulary, and scaler."""
    if not (os.path.exists('model.pkl') and os.path.exists('voc.pkl') and os.path.exists('scaler.pkl')):
        raise FileNotFoundError("Missing one or more model files. Please run train.py first.")
        
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('voc.pkl', 'rb') as f:
        voc = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, voc, scaler

def predict_signature(test_image_path):
    """
    Predict whether a signature is Genuine or Forged.
    Uses pre-computed vocabulary and scaler for fast O(1) inference.
    """
    try:
        model, voc, scaler = load_model_components()
    except Exception as e:
        raise Exception(f"Failed to load model components: {e}")

    geom, des, valid = extract_features(test_image_path)
    if not valid or des is None or len(des) == 0:
        raise Exception("Failed to extract features from the image. It might be corrupt or lack distinct features.")

    k = voc.shape[0]
    
    # Generate histogram features
    im_features = np.zeros((1, k + 12), "float32")
    words, distance = vq(des, voc)
    for w in words:
        if w < k:
            im_features[0][w] += 1
            
    # Append 12 geometric features
    for j in range(12):
        im_features[0][k + j] = geom[j]
        
    # Scale features
    im_features_scaled = scaler.transform(im_features)
    
    # Predict
    # 2 -> Genuine, 1 -> Forged
    prediction = model.predict(im_features_scaled)[0]
    
    # For SVC linear we can get decision function distance instead of probab
    decision_val = model.decision_function(im_features_scaled)[0]
    
    # Approximate confidence from decision value
    dist = abs(decision_val)
    # Basic clamping and scaling for visualization
    conf_score = min(99.0, max(50.0, 50.0 + (dist * 10.0)))
    
    result = "Genuine" if prediction == 2 else "Forged"
    
    confidence = {
        "genuine_accuracy": conf_score if prediction == 2 else 100 - conf_score,
        "forged_accuracy": conf_score if prediction == 1 else 100 - conf_score
    }
    
    return result, confidence
