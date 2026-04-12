import numpy as np
import os
import cv2
from PIL import Image
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
import imagehash
import sys

import preproc
import features

def extract_features(image_path, sift_detector):
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
        
        geom_features = [
            aspect_ratio, 
            convex_hull_area/bounding_rect_area if bounding_rect_area > 0 else 0,
            contours_area/bounding_rect_area if bounding_rect_area > 0 else 0,
            ratio, centroid_0, centroid_1, eccentricity, solidity,
            skewness_0, skewness_1, kurtosis_0, kurtosis_1
        ]
        
        _, des = sift_detector.detectAndCompute(prep_img, None)
        return geom_features, des, True
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None, None, False

def train_model():
    print("Initializing feature extraction...")
    try:
        sift_detector = cv2.xfeatures2d.SIFT_create()
    except AttributeError:
        sift_detector = cv2.SIFT_create()

    genuine_dir = "data/genuine"
    forged_dir = "data/forged"
    
    if not os.path.exists(genuine_dir) or not os.path.exists(forged_dir):
        print("Data directories not found. Please ensure data/genuine and data/forged exist.")
        return

    genuine_files = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir) if f.endswith('.png') or f.endswith('.jpg')]
    forged_files = [os.path.join(forged_dir, f) for f in os.listdir(forged_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    all_files = genuine_files + forged_files
    labels = [2]*len(genuine_files) + [1]*len(forged_files)  # 2 for genuine, 1 for forged
    
    all_geom_features = []
    all_descriptors = []
    valid_indices = []
    
    print("Extracting features from images...")
    for idx, filepath in enumerate(all_files):
        geom, des, valid = extract_features(filepath, sift_detector)
        if valid and des is not None and len(des) > 0:
            all_geom_features.append(geom)
            all_descriptors.append(des)
            valid_indices.append(idx)
            
        if (idx+1) % 20 == 0:
            print(f"Processed {idx+1}/{len(all_files)} images...")

    valid_labels = [labels[i] for i in valid_indices]
    
    print("Formatting SIFT descriptors...")
    # Stack all descriptors for K-Means
    stacked_descriptors = all_descriptors[0]
    for des in all_descriptors[1:]:
        stacked_descriptors = np.vstack((stacked_descriptors, des))
    
    print(f"Total descriptors found: {stacked_descriptors.shape[0]}")
    k = min(500, stacked_descriptors.shape[0])
    
    print(f"Running K-Means clustering (k={k}). This may take a moment...")
    voc, variance = kmeans(stacked_descriptors, k, 1)
    
    print("Generating histogram features...")
    n_images = len(all_geom_features)
    im_features = np.zeros((n_images, k + 12), "float32")
    
    for i in range(n_images):
        words, distance = vq(all_descriptors[i], voc)
        for w in words:
            if w < k:
                im_features[i][w] += 1
        
        # Append 12 geometric features
        for j in range(12):
            im_features[i][k + j] = all_geom_features[i][j]
            
    print("Scaling features...")
    scaler = StandardScaler().fit(im_features)
    im_features_scaled = scaler.transform(im_features)
    
    print("Training LinearSVC SVM model...")
    clf = LinearSVC(max_iter=10000)
    clf.fit(im_features_scaled, valid_labels)
    
    train_acc = clf.score(im_features_scaled, valid_labels)
    print(f"Training completed! Accuracy on training set: {train_acc * 100:.2f}%")
    
    print("Saving models to disk...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('voc.pkl', 'wb') as f:
        pickle.dump(voc, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("All necessary files (model.pkl, voc.pkl, scaler.pkl) saved successfully.")

if __name__ == "__main__":
    train_model()
