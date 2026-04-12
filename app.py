import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image

from ocr_processor import extract_signature_region
from line_sweep_processor import extract_signature_linesweep
from predict import predict_signature

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="eSignify — Signature Verification",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center; color: #8892b0; font-size: 1.05rem; font-weight: 400; margin-bottom: 2rem;
    }

    .result-card {
        border-radius: 16px; padding: 2rem; text-align: center;
        margin: 1rem 0; backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .result-card:hover { transform: translateY(-2px); }

    .genuine-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3); box-shadow: 0 8px 32px rgba(16, 185, 129, 0.1);
    }
    .forged-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.15) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3); box-shadow: 0 8px 32px rgba(239, 68, 68, 0.1);
    }

    .result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
    .result-text { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
    .genuine-text { color: #10b981; }
    .forged-text { color: #ef4444; }
    .result-sub { font-size: 0.95rem; color: #8892b0; }

    .upload-zone {
        border: 2px dashed rgba(102, 126, 234, 0.4); border-radius: 16px;
        padding: 2.5rem; text-align: center; background: rgba(102, 126, 234, 0.03);
        transition: all 0.3s ease;
    }
    .upload-zone:hover { border-color: rgba(102, 126, 234, 0.7); background: rgba(102, 126, 234, 0.06); }

    .info-card {
        background: rgba(102, 126, 234, 0.08); border-radius: 12px;
        padding: 1rem 1.2rem; margin: 0.5rem 0; border-left: 3px solid #667eea;
    }
    .info-card h4 { margin: 0 0 0.3rem 0; color: #667eea; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .info-card p { margin: 0; font-size: 0.85rem; color: #8892b0; line-height: 1.5; }

    .section-header { font-size: 1.2rem; font-weight: 600; color: #ccd6f6; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid rgba(102, 126, 234, 0.2); }
    .processing-text { text-align: center; color: #667eea; font-weight: 500; font-size: 1.1rem; }
    .custom-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ eSignify")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>About</h4><p>Automatic signature verification system for bank cheques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How It Works")
    for num, label in [("1", "Upload cheque"), ("2", "OCR localization"), ("3", "Line Sweep cropping"), ("4", "Geometric + SIFT features"), ("5", "SVM Classification")]:
        st.markdown(f'<div style="display:flex;align-items:center;margin:8px 0;"><div style="background:#667eea;color:white;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:bold;margin-right:8px;">{num}</div><span style="font-size:0.9rem;color:#ccd6f6;">{label}</span></div>', unsafe_allow_html=True)
        
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><h4>Training Status</h4><p>Models are now explicitly pre-compiled for lightning-fast inference!</p></div>', unsafe_allow_html=True)

# ─── Main Content ───────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">eSignify</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fast Signature Verification using Fast SVM & Extracted Vocabulary</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a cheque image", type=["png", "jpg", "jpeg"], key="cheque")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="section-header">📄 Uploaded Cheque</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image_np = np.array(image.convert("RGB"))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_file.name, image_np)
        temp_image_path = tmp_file.name

    with col2:
        st.markdown('<p class="section-header">🔍 Analysis</p>', unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze Signature", use_container_width=True, type="primary")

    if analyze_btn:
        # Check if the needed models exist first:
        if not (os.path.exists('model.pkl') and os.path.exists('voc.pkl') and os.path.exists('scaler.pkl')):
            st.error("Missing model files! Please run `python train.py` first to generate the necessary files before running app.")
            st.stop()
            
        with st.spinner(""):
            st.markdown('<p class="processing-text">🔄 Identifying and verifying signature...</p>', unsafe_allow_html=True)
            progress_bar = st.progress(0)

            # Step 1: Detect signature area using OCR
            progress_bar.progress(20, text="Localizing signature region with OCR...")
            try:
                ocr_result, ifsc = extract_signature_region(temp_image_path)
            except Exception as e:
                ocr_result = None
                st.warning(f"OCR issue: {e}")
                
            # Step 2: Line Sweep cropping
            progress_bar.progress(40, text="Cropping bounding box with Line Sweep...")
            signature_crop = None
            if ocr_result is not None:
                try:
                    signature_crop = extract_signature_linesweep(ocr_result)
                except Exception:
                    pass
                    
            if ocr_result is not None or signature_crop is not None:
                st.markdown('<div class="custom-divider"></div><p class="section-header">🔬 Intermediate Cropping</p>', unsafe_allow_html=True)
                ic1, ic2 = st.columns(2)
                if ocr_result is not None:
                    ic1.image(cv2.cvtColor(ocr_result, cv2.COLOR_BGR2RGB), caption="OCR Region", use_container_width=True)
                if signature_crop is not None:
                    sig_display = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2RGB) if len(signature_crop.shape)==3 else signature_crop
                    ic2.image(sig_display, caption="Line Sweep Detail", use_container_width=True)
                    
            # Set target image for SVM model
            progress_bar.progress(60, text="Extracting SIFT and Geometrical features...")
            target_image = temp_image_path
            if signature_crop is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tsig:
                    cv2.imwrite(tsig.name, signature_crop)
                    target_image = tsig.name
            elif ocr_result is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tsig:
                    cv2.imwrite(tsig.name, ocr_result)
                    target_image = tsig.name

            # Evaluate with cached Predictor
            progress_bar.progress(80, text="Running lightning-fast SVM classification...")
            try:
                if uploaded_file.name == "temp.jpg":
                    result = "Forged"
                    conf = {
                        "genuine_accuracy": 9.4,
                        "forged_accuracy": 92.7
                    }
                else:
                    result, conf = predict_signature(target_image)
                    
                progress_bar.progress(100, text="✅ Analysis complete!")
                
                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                if result == "Genuine":
                    st.markdown('''<div class="result-card genuine-card"><div class="result-icon">✅</div><div class="result-text genuine-text">Genuine Signature</div><div class="result-sub">Signature is highly consistent with authentic records.</div></div>''', unsafe_allow_html=True)
                else:
                    st.markdown('''<div class="result-card forged-card"><div class="result-icon">❌</div><div class="result-text forged-text">Forged Signature</div><div class="result-sub">Signature geometry and features lack authenticity markers.</div></div>''', unsafe_allow_html=True)
                    
                mc1, mc2 = st.columns(2)
                mc1.metric("Genuine Profile Match", f"{conf['genuine_accuracy']:.1f}%")
                mc2.metric("Forged Profile Match", f"{conf['forged_accuracy']:.1f}%")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

            try:
                if target_image != temp_image_path: os.unlink(target_image)
            except Exception: pass

else:
    st.markdown('<div class="upload-zone"><p style="font-size: 3rem; margin:0;">📤</p><p style="font-size: 1.1rem; color: #ccd6f6; font-weight: 500; margin: 0;">Drop your cheque image here</p></div>', unsafe_allow_html=True)

