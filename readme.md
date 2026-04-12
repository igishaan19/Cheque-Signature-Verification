# eSignify — Signature Verification on Bank Cheques

A Streamlit application for detecting forged signatures on bank cheques using computer vision and machine learning.

## Overview

eSignify uses OCR-based signature detection combined with SVM classification to verify whether a signature on a bank cheque is **genuine** or **forged**. The system extracts geometric and SIFT features from the signature and compares against a database of known genuine and forged signatures.

## Features

- **Signature Detection**: Automatic signature localization using Tesseract OCR + Line Sweep algorithm
- **Feature Extraction**: SIFT descriptors + geometric features (aspect ratio, centroid, eccentricity, skewness, kurtosis)
- **SVM Classification**: LinearSVC-based genuine/forged classification with ~91% accuracy
- **Interactive UI**: Clean Streamlit interface with real-time processing feedback

## Project Structure

```
eSignify/
├── app.py                    # Streamlit application (main entry point)
├── ocr_processor.py          # OCR-based signature region detection
├── line_sweep_processor.py   # Line Sweep signature cropping
├── predict.py                # SVM prediction pipeline
├── preproc.py                # Image preprocessing (RGB → grayscale → binary)
├── features.py               # Geometric feature extraction
├── model.pkl                 # Pre-trained SVM model
├── data/
│   ├── genuine/              # Reference genuine signatures (29 users × 5 samples)
│   └── forged/               # Reference forged signatures (29 users × 5 samples)
├── requirements.txt          # Python dependencies
└── readme.md                 # This file
```

## Prerequisites

- **Python** 3.8+
- **Tesseract OCR** installed and accessible
  - Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
  - Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd eSignify
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR** (if not already installed):
   - Windows: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)
   - Ensure it's in your system PATH or at `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then:
1. Upload a scanned cheque image (PNG, JPG, or JPEG)
2. Click **"🚀 Analyze Signature"**
3. View the detection steps and final result (Genuine ✅ or Forged ❌)

## How It Works

1. **OCR Detection**: Tesseract identifies text on the cheque to locate the "Please Sign Above" region
2. **Line Sweep**: The algorithm crops the signature tightly using horizontal and vertical sweeps
3. **Feature Extraction**: SIFT descriptors and geometric features are computed
4. **SVM Classification**: A LinearSVC model compares the test signature against the reference database
