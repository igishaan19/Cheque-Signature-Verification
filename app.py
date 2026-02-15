import streamlit as st
import cv2
import numpy as np
import joblib
from extract import extract_signature

st.set_page_config(page_title="Signature Verification System")

st.title("Signature Verification System")

st.write("Prototype interface - Work in progress")

uploaded_file = st.file_uploader("Upload Cheque Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Cheque", use_container_width=True)
    st.success("Cheque uploaded successfully!")


if uploaded_file is not None:
    if st.button("Extract Signature"):
        st.info("Extracting signature... (prototype)")


model = st.selectbox("Choose Verification Model", ["SVM", "Random Forest", "CNN (Coming Soon)"])
st.write("Selected model:", model)

import time

if st.button("Start Verification"):

    if uploaded_file is not None:

        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        cv2.imwrite("temp.jpg", image)

        signature = extract_signature("temp.jpg")

        if signature is None:
            st.error("No signature detected")
        else:
            gray = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
            area = np.count_nonzero(gray)
            perimeter = gray.shape[0] + gray.shape[1]
            aspect_ratio = gray.shape[1] / gray.shape[0]
            edge_pixels = np.count_nonzero(gray)
            contour_ratio = area / (gray.shape[0] * gray.shape[1])

            features = [[area, perimeter, aspect_ratio, edge_pixels, contour_ratio]]

            model = joblib.load("models/signature_model.pkl")
            prediction = model.predict(features)[0]
            signature_rgb = cv2.cvtColor(signature, cv2.COLOR_BGR2RGB)
            st.image(signature_rgb, caption="Extracted Signature", use_container_width=True)


            if prediction == 1:
                st.success("Prediction: Genuine Signature")
            else:
                st.error("Prediction: Forged Signature")

st.markdown("---")
st.subheader("About the Project")
st.write("This is a prototype Signature Verification System developed for PBL.")



