import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PneumoDetect - Pneumonia AI Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - FIXED FOR DARK THEME
# CRITICAL: Light text on dark backgrounds
# ============================================================

st.markdown("""
    <style>
    /* ROOT COLOR VARIABLES */
    :root {
        --primary: #0066cc;
        --primary-dark: #0047a3;
        --success: #28a745;
        --danger: #dc3545;
        --warning: #ff9800;
        --light: #f8f9fa;
        --dark: #333333;
    }
    
    /* FORCE ALL TEXT TO BE VISIBLE */
    /* Override Streamlit's default dark theme */
    
    /* Light mode text */
    p, span, li, label {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    .stMarkdown {
        color: #ffffff !important;
    }
    
    .stMarkdown p {
        color: #ffffff !important;
    }
    
    /* All divs must have light text */
    div {
        color: #ffffff !important;
    }
    
    /* ============================
       HEADER STYLING
       ============================ */
    
    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #0047a3 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.2);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin: 0;
        font-weight: 700;
        color: #ffffff !important;
    }
    
    .main-header p {
        font-size: 1.1em;
        margin: 10px 0 0 0;
        opacity: 0.95;
        color: #ffffff !important;
    }
    
    /* ============================
       STATISTICS CARDS
       ============================ */
    
    .stat-card {
        background: #1a1a2e;
        border: 2px solid #16213e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.3);
        transform: translateY(-2px);
        border-color: #0066cc;
    }
    
    .stat-number {
        font-size: 2em;
        font-weight: 700;
        color: #60a5fa !important;
    }
    
    .stat-label {
        font-size: 0.95em;
        color: #e0e0e0 !important;
        margin-top: 5px;
    }
    
    /* ============================
       RESULT BOXES
       ============================ */
    
    .result-positive {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3);
    }
    
    .result-positive h2,
    .result-positive div,
    .result-positive p {
        color: #ffffff !important;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(22, 163, 74, 0.3);
    }
    
    .result-negative h2,
    .result-negative div,
    .result-negative p {
        color: #ffffff !important;
    }
    
    .result-title {
        font-size: 2em;
        font-weight: 700;
        margin-bottom: 10px;
        color: #ffffff !important;
    }
    
    .result-confidence {
        font-size: 1.5em;
        margin: 15px 0;
        color: #ffffff !important;
    }
    
    /* ============================
       INFO / WARNING / SUCCESS BOXES
       ============================ */
    
    .info-box {
        background: #1e3a8a;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        border-radius: 6px;
        margin: 15px 0;
        color: #e0e7ff !important;
    }
    
    .info-box b,
    .info-box strong {
        color: #60a5fa !important;
    }
    
    .info-box p,
    .info-box span,
    .info-box div {
        color: #e0e7ff !important;
    }
    
    .warning-box {
        background: #78350f;
        border-left: 5px solid #facc15;
        padding: 15px;
        border-radius: 6px;
        margin: 15px 0;
        color: #fef3c7 !important;
    }
    
    .warning-box b,
    .warning-box strong {
        color: #fcd34d !important;
    }
    
    .warning-box p,
    .warning-box span,
    .warning-box div {
        color: #fef3c7 !important;
    }
    
    .success-box {
        background: #064e3b;
        border-left: 5px solid #22c55e;
        padding: 15px;
        border-radius: 6px;
        margin: 15px 0;
        color: #dcfce7 !important;
    }
    
    .success-box b,
    .success-box strong {
        color: #86efac !important;
    }
    
    .success-box p,
    .success-box span,
    .success-box div {
        color: #dcfce7 !important;
    }
    
    /* ============================
       TAB STYLING
       ============================ */
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #d1d5db;
        background-color: #1f2937;
        border-bottom: 2px solid #374151;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #ffffff !important;
        background-color: #0066cc !important;
        border-bottom: 2px solid #0047a3;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] span,
    .stTabs [data-baseweb="tab-panel"] div {
        color: #ffffff !important;
    }
    
    /* ============================
       UPLOAD AREA
       ============================ */
    
    .upload-area {
        border: 2px dashed #0066cc;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background: #1f2937;
        margin: 20px 0;
        color: #ffffff !important;
    }
    
    /* ============================
       BUTTONS
       ============================ */
    
    .stButton button {
        background-color: #0066cc !important;
        color: #ffffff !important;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 24px;
    }
    
    .stButton button:hover {
        background-color: #0047a3 !important;
    }
    
    /* ============================
       SIDEBAR
       ============================ */
    
    section[data-testid="stSidebar"] {
        background-color: #1f2937;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #60a5fa !important;
    }
    
    /* ============================
       FOOTER
       ============================ */
    
    .footer-section {
        text-align: center;
        padding: 15px;
        color: #d1d5db !important;
        background: #1f2937;
        border-radius: 6px;
    }
    
    .footer-section b,
    .footer-section strong {
        color: #ffffff !important;
    }
    
    .footer-section p,
    .footer-section span,
    .footer-section div {
        color: #ffffff !important;
    }
    
    .footer-box {
        text-align: center;
        margin-top: 20px;
        padding: 15px;
        background: #111827;
        border-radius: 6px;
        color: #9ca3af !important;
        border: 1px solid #374151;
    }
    
    .footer-box small {
        color: #9ca3af !important;
    }
    
    /* ============================
       STREAMLIT NATIVE OVERRIDES
       ============================ */
    
    .stInfo {
        background-color: #1e3a8a !important;
        border-left: 5px solid #3b82f6 !important;
    }
    
    .stInfo p,
    .stInfo span,
    .stInfo div {
        color: #e0e7ff !important;
    }
    
    .stSuccess {
        background-color: #064e3b !important;
        border-left: 5px solid #22c55e !important;
    }
    
    .stSuccess p,
    .stSuccess span,
    .stSuccess div {
        color: #dcfce7 !important;
    }
    
    .stError {
        background-color: #7f1d1d !important;
        border-left: 5px solid #ef4444 !important;
    }
    
    .stError p,
    .stError span,
    .stError div {
        color: #fee2e2 !important;
    }
    
    .stWarning {
        background-color: #78350f !important;
        border-left: 5px solid #facc15 !important;
    }
    
    .stWarning p,
    .stWarning span,
    .stWarning div {
        color: #fef3c7 !important;
    }
    
    /* ============================
       MARKDOWN TEXT FIX
       ============================ */
    
    .element-container,
    .block-container {
        color: #ffffff !important;
    }
    
    .element-container p {
        color: #ffffff !important;
    }
    
    /* List items */
    li {
        color: #ffffff !important;
    }
    
    /* Strong and bold text */
    b, strong {
        color: #ffffff !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
MODEL_PATH = "chest_xray_model_fixed.keras"

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained pneumonia detection model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):
    """Convert and prepare image for model prediction"""
    try:
        image = image.convert("L")  # Grayscale
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.array(image)
        # Convert grayscale to RGB
        image_array = np.stack((image_array,) * 3, axis=-1)
        # ResNet50 preprocessing
        image_array = preprocess_input(image_array)
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

# ============================================================
# MAIN APP HEADER
# ============================================================

st.markdown("""
    <div class="main-header">
        <h1>🫁 PneumoDetect</h1>
        <p>AI-Powered Pneumonia Detection from Chest X-rays</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 📋 Navigation")
    page = st.radio("Select a page:", 
        ["🏠 Home", "📊 Statistics", "📖 About Pneumonia", "🔬 Detector"])

# ============================================================
# PAGE 1: HOME
# ============================================================

if page == "🏠 Home":
    st.markdown("""
    ### Welcome to PneumoDetect
    
    PneumoDetect is an advanced AI system designed to assist radiologists and healthcare professionals 
    in detecting pneumonia from chest X-ray images using deep learning technology.
    
    **Key Features:**
    - 🤖 State-of-the-art ResNet50 neural network
    - ⚡ Real-time detection with high accuracy
    - 📈 Comprehensive statistics and insights
    - 🩺 Medical information and guidance
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">86%</div>
            <div class="stat-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number"><1s</div>
            <div class="stat-label">Detection Speed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">99.9%</div>
            <div class="stat-label">Uptime</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Quick Start
    
    1. **Go to the Detector tab** (🔬 Detector in the sidebar)
    2. **Upload a chest X-ray image** (JPG, PNG, or JPEG format)
    3. **Click Analyze** to get instant results
    4. **Review recommendations** based on the diagnosis
    
    ### ⚠️ Important Disclaimer
    
    **This tool is for educational and assistive purposes only.** 
    - It should NOT be used as a standalone diagnostic tool
    - Always consult qualified healthcare professionals
    - Results must be verified by certified radiologists
    - Patient care decisions should be made by medical experts
    """)

# ============================================================
# PAGE 2: STATISTICS
# ============================================================

elif page == "📊 Statistics":
    st.title("📊 Pneumonia Statistics & Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Global Pneumonia Burden
        
        **Epidemiology:**
        - ~450 million pneumonia cases annually worldwide
        - ~4 million deaths per year
        - Leading infectious cause of death globally
        - Affects all age groups, but children <5 and adults >65 at highest risk
        
        **Economic Impact:**
        - Over $100 billion in healthcare costs annually
        - Increased hospitalization and productivity loss
        - Most common reason for hospital admission
        """)
    
    with col2:
        # Create a chart
        causes = ["Bacterial", "Viral", "Fungal", "Atypical"]
        percentages = [50, 30, 10, 10]
        
        fig = go.Figure(data=[go.Pie(
            labels=causes,
            values=percentages,
            marker=dict(colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']),
            textposition='inside',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title="Pneumonia Causes Distribution",
            height=400,
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">450M</div>
            <div class="stat-label">Annual Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">4M</div>
            <div class="stat-label">Annual Deaths</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">1-2%</div>
            <div class="stat-label">Mortality (Treated)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Risk Factors
    
    **High-Risk Groups:**
    - Children under 5 years old
    - Adults over 65 years
    - Immunocompromised individuals
    - Smokers
    - People with chronic diseases (COPD, asthma, diabetes)
    - Recent surgeries or prolonged bed rest
    """)

# ============================================================
# PAGE 3: ABOUT PNEUMONIA
# ============================================================

elif page == "📖 About Pneumonia":
    st.title("📖 Understanding Pneumonia")
    
    st.markdown("""
    ### What is Pneumonia?
    
    Pneumonia is an infection that inflames the lung's air sacs (alveoli), causing them to fill with 
    fluid or pus. This makes it difficult for oxygen to reach the bloodstream and impairs the body's 
    ability to fight the infection.
    """)
    
    st.markdown("---")
    
    st.markdown("### Types of Pneumonia")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Bacterial", "Viral", "Fungal", "Atypical"])
    
    with tab1:
        st.markdown("""
        **Bacterial Pneumonia**
        - Most common type (~30% of cases)
        - Caused by bacteria like *Streptococcus pneumoniae*
        - Often sudden onset
        - Generally responds well to antibiotics
        - More severe if untreated
        """)
    
    with tab2:
        st.markdown("""
        **Viral Pneumonia**
        - Most common type (~40% of cases)
        - Caused by viruses like RSV, influenza, COVID-19
        - Often milder but can progress
        - Usually self-limiting
        - May develop secondary bacterial infection
        """)
    
    with tab3:
        st.markdown("""
        **Fungal Pneumonia**
        - Less common (~10% of cases)
        - Often in immunocompromised patients
        - Caused by fungi like Aspergillus, Candida
        - Difficult to diagnose
        - Requires antifungal treatment
        """)
    
    with tab4:
        st.markdown("""
        **Atypical Pneumonia**
        - About 10% of cases
        - Caused by organisms like Mycoplasma, Legionella
        - Often called "walking pneumonia"
        - May have unusual symptoms
        - Responds to specific antibiotics
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Symptoms
    
    **Common Symptoms:**
    - 🤒 Cough (often productive with sputum)
    - 🌡️ Fever and chills
    - 💨 Shortness of breath or difficulty breathing
    - 🫀 Chest pain when breathing or coughing
    - 😴 Fatigue and weakness
    - 🤢 Nausea or diarrhea
    
    **Seek Emergency Care If:**
    - Difficulty breathing or severe shortness of breath
    - Chest pain
    - Confusion or altered mental state
    - Blue lips or fingertips (cyanosis)
    - Severe weakness
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Diagnosis & Treatment
    
    **Diagnostic Methods:**
    - Chest X-ray (primary imaging method)
    - CT scans (for complex cases)
    - Blood tests and cultures
    - Sputum analysis
    
    **Treatment:**
    - **Antibiotics** for bacterial pneumonia
    - **Antiviral drugs** for severe viral pneumonia
    - **Oxygen therapy** for severe cases
    - **Supportive care** (rest, fluids, nutrition)
    - **Hospitalization** for severe cases
    
    **Prevention:**
    - 💉 Pneumococcal vaccine
    - 💉 Influenza vaccination
    - 🚭 Avoid smoking
    - 🧼 Good hand hygiene
    - 😷 Mask use in high-risk environments
    """)

# ============================================================
# PAGE 4: DETECTOR
# ============================================================

elif page == "🔬 Detector":
    st.title("🔬 Pneumonia Detection System")
    
    st.markdown("""
    <div class="info-box">
        <b>📌 How to use:</b> Upload a chest X-ray image in JPG, PNG, or JPEG format. 
        The AI model will analyze it and provide a diagnosis with confidence score.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please check the model file.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    st.markdown("---")
    
    # File upload
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload Chest X-ray Image",
            type=["jpg", "jpeg", "png"],
            help="Select a chest X-ray image for analysis"
        )
    
    with col2:
        st.markdown("")  # Spacing
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖼️ Uploaded X-ray")
            st.image(image, use_container_width=True, output_format="PNG")
        
        with col2:
            st.markdown("### 📋 Image Information")
            st.info(f"""
            - **Filename:** {uploaded_file.name}
            - **Size:** {image.size[0]} × {image.size[1]} pixels
            - **Format:** {image.format}
            """)
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🔍 Analyze X-ray", use_container_width=True, type="primary"):
            with st.spinner("🤖 Analyzing image with AI model..."):
                # Preprocess and predict
                processed_img = preprocess_image(image)
                
                if processed_img is not None:
                    prediction = model.predict(processed_img, verbose=0)
                    prob = float(prediction[0][0])
                    confidence = prob if prob > 0.5 else (1 - prob)
                    
                    st.markdown("---")
                    
                    # Display result
                    if prob > 0.5:
                        # PNEUMONIA DETECTED
                        st.markdown("""
                        <div class="result-positive">
                            <div class="result-title">⚠️ PNEUMONIA DETECTED</div>
                            <div class="result-confidence">Confidence: {:.1f}%</div>
                        </div>
                        """.format(confidence * 100), unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="warning-box">
                            <b>⚠️ Important Notice:</b><br>
                            This analysis suggests pneumonia may be present. Immediate consultation 
                            with a healthcare professional is recommended.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Actions for positive result
                        st.markdown("### 🏥 Recommended Actions")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            **Immediate Steps:**
                            1. ☎️ **Contact your doctor** immediately
                            2. 🏥 **Schedule urgent appointment** or visit ER if severe
                            3. 📸 **Share this result** with healthcare provider
                            4. 📝 **Prepare medical history**
                            """)
                        
                        with col2:
                            st.markdown("""
                            **Questions to Ask Doctor:**
                            - What type of pneumonia is it?
                            - What is the recommended treatment?
                            - Do I need antibiotics?
                            - Should I be hospitalized?
                            - What complications should I watch for?
                            """)
                        
                        st.markdown("---")
                        
                        st.markdown("""
                        ### ⚠️ Warning Signs (Seek Emergency Care)
                        
                        Go to the emergency room immediately if you experience:
                        - 💨 Severe difficulty breathing
                        - 💙 Bluish lips or fingernails
                        - 😵 Confusion or loss of consciousness
                        - 💔 Chest pain with each breath
                        - 🌡️ Fever above 40°C (104°F)
                        """)
                        
                    else:
                        # NORMAL
                        st.markdown("""
                        <div class="result-negative">
                            <div class="result-title">✅ NORMAL</div>
                            <div class="result-confidence">Confidence: {:.1f}%</div>
                        </div>
                        """.format(confidence * 100), unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="success-box">
                            <b>✅ Good News:</b><br>
                            This X-ray does not show signs of pneumonia. However, continue monitoring 
                            your health and consult a doctor if symptoms persist.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Actions for negative result
                        st.markdown("### 💚 Health Recommendations")
                        
                        st.markdown("""
                        **Maintain Good Health:**
                        - 💪 Stay active with regular exercise
                        - 🥗 Eat a balanced, nutritious diet
                        - 💤 Get 7-9 hours of quality sleep
                        - 🚭 Avoid smoking and secondhand smoke
                        - 🧼 Practice good hygiene
                        
                        **Prevention Tips:**
                        - 💉 Stay up-to-date with vaccinations
                        - 🧴 Use hand sanitizer regularly
                        - 😷 Wear masks in high-risk environments
                        - 🤝 Limit exposure to sick people
                        - 🧠 Manage stress effectively
                        """)
                        
                        st.markdown("---")
                        
                        st.info("""
                        **Note:** Even with a normal result, if you have persistent respiratory 
                        symptoms, consult your healthcare provider for further evaluation.
                        """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="footer-section">
        <p><b>🏥 About This Tool</b><br>
        Deep learning-powered pneumonia detection using advanced neural networks</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="footer-section">
        <p><b>⚖️ Disclaimer</b><br>
        For educational use only. Not a replacement for professional diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="footer-section">
        <p><b>📧 Support</b><br>
        For questions, contact healthcare professionals.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-box">
    <small>© 2025 PneumoDetect. All rights reserved. | Built with ❤️ for healthcare</small>
</div>
""", unsafe_allow_html=True)
