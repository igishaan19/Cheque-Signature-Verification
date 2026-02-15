# Cheque Signature Verification System

# Overview
This project implements a machine learning based system to verify handwritten signatures on bank cheques. The goal is to reduce fraud in financial institutions by automatically classifying signatures as genuine or forged using image processing and classification techniques.

---

# Problem Statement
Manual signature verification is time-consuming and prone to human error. There is a need for an automated system that can accurately verify signatures and reduce cheque fraud.

---

# Objectives
- To preprocess and enhance signature images
- To extract meaningful geometric features
- To train a machine learning classifier for verification
- To classify signatures as genuine or forged

---

# Methodology
1. Load signature image
2. Convert image to grayscale
3. Apply thresholding for segmentation
4. Extract features (area, perimeter, aspect ratio)
5. Train Support Vector Machine (SVM)
6. Predict authenticity of signature

---

# Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

# Expected Outcome
The system is expected to achieve approximately 85–92% accuracy depending on dataset quality.

---

# Future Scope
- Integration with real-time cheque scanning systems
- Improvement using deep learning models
- Deployment as a banking application module

---

# Author
Ishaan Gupta  
Manipal University Jaipur  
PBL Project 2026
