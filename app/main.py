import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import numpy as np

def getCleanData():
    data = pd.read_csv("data.csv")

    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data

def addSidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = getCleanData()

    sliderLabels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    inputDict = {}
    for label, key in sliderLabels:
        inputDict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return inputDict    

def scaledValues(inputDict):
    data = getCleanData()

    X = data.drop(['diagnosis'], axis=1)
    scaledDict = {}

    for key, value in inputDict.items():
        maxVal = X[key].max()
        minVal = X[key].min()
        scaledVal = (value-minVal)/(maxVal-minVal)
        scaledDict[key] = scaledVal
    return scaledDict


def getRadarChart(inputData):
    inputData=scaledValues(inputData)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concativity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            inputData['radius_mean'], inputData['texture_mean'], inputData['perimeter_mean'],
            inputData['area_mean'], inputData['smoothness_mean'], inputData['compactness_mean'], 
            inputData['concavity_mean'], inputData['concave points_mean'], inputData['symmetry_mean'], 
            inputData['fractal_dimension_mean'] 
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          inputData['radius_se'], inputData['texture_se'], inputData['perimeter_se'], inputData['area_se'],
          inputData['smoothness_se'], inputData['compactness_se'], inputData['concavity_se'],
          inputData['concave points_se'], inputData['symmetry_se'],inputData['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          inputData['radius_worst'], inputData['texture_worst'], inputData['perimeter_worst'],
          inputData['area_worst'], inputData['smoothness_worst'], inputData['compactness_worst'],
          inputData['concavity_worst'], inputData['concave points_worst'], inputData['symmetry_worst'],
          inputData['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )
    return fig

def addPredictions(inputData):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    inputArray = np.array(list(inputData.values())).reshape(1,-1)
    inputArrayScaled = scaler.transform(inputArray)
    prediction = model.predict(inputArrayScaled)
    
    st.subheader("Cell Cluster Prediction")
    st.write("The Cell Cluster is: ")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    st.write("Probability of being Benign: ", model.predict_proba(inputArrayScaled)[0][0])
    st.write("Probability of being Malicious: ", model.predict_proba(inputArrayScaled)[0][1])
    st.write("This app is to assist professionals. It is not a substitute for a professional diagnosis from a lab")

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
        )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    inputData = addSidebar()
    

    with st.container():
        st.write('<span class="title">Breast Cancer Predictor</span>', unsafe_allow_html=True)
        st.write("Provides automated analysis of breast mass features based on input values using Machine Learning.")
    
    col1, col2 = st.columns([4,1])
    with col1:
        radarChart=getRadarChart(inputData)
        st.plotly_chart(radarChart)
    with col2:
        addPredictions(inputData)

    # Title and Introduction
    st.header("🌡️ Understanding Breast Cancer")
    st.write("""
    Breast cancer is a disease where cells in the breast grow out of control, often forming a tumor. 
    It's the most common cancer in women worldwide, though it can also affect men. 
    *Early detection through screening and regular self-exams significantly improves outcomes.*
    """)

    # Main content columns - renamed to avoid conflicts
    info_col1, info_col2 = st.columns([1, 1])

    with info_col1:
        # What is Breast Cancer?
        with st.expander("🔬 What is Breast Cancer?", expanded=True):
            st.markdown("""
            - **Abnormal cell growth** in breast tissue
            - Most common in **milk ducts** (ductal carcinoma) or **lobules** (lobular carcinoma)
            - Can spread (**metastasize**) if malignant
            - Many types with different growth rates
            """)
        
        # Key Facts and Statistics
        with st.expander("📊 Key Facts & Statistics"):
            st.markdown("""
            **Global Impact (2022):**
            - 🗺️ Most common cancer in women (157 countries)
            - ⚠️ 670,000 annual deaths worldwide
            
            **Demographics:**
            - 👩 More frequent in women
            - 👨 Can occur in men (1% of cases)
            - 🎂 Risk increases with age
            
            **Risk Factors:**
            - 🧬 BRCA1/BRCA2 gene mutations
            - 👪 Family history
            - 🍷 Lifestyle factors
            """)

    with info_col2:
        # Signs and Symptoms
        with st.expander("⚠️ Signs & Symptoms"):
            st.markdown("""
            **Physical Changes:**
            - 🔍 New lump/mass (often painless)
            - 🏐 Swelling in breast/armpit
            - 🍊 Skin dimpling (like orange peel)
            
            **Nipple Changes:**
            - ↩️ Inversion
            - 🩹 Unusual discharge
            - 🔴 Redness/scaling
            
            *Note:* 80% of lumps are benign, but **always get checked**.
            """)
        
        # Stages
        with st.expander("📈 Cancer Stages"):
            st.markdown("""
            **Stage 0:** Non-invasive (DCIS)  
            **Stage I:** Small tumor, localized  
            **Stage II:** Larger or lymph node involvement  
            **Stage III:** Spread to nearby tissues  
            **Stage IV:** Metastasized to other organs  
            
            Staging determines treatment options.
            """)

    # Full-width sections
    st.subheader("🩺 Screening Methods")
    screen_col1, screen_col2, screen_col3 = st.columns(3)
    with screen_col1:
        st.markdown("**Mammogram**\n\nX-ray screening\n(Recommended annually after 40)")
    with screen_col2:
        st.markdown("**Clinical Exam**\n\nDoctor's physical check\n(Every 3 years for 20s-30s)")
    with screen_col3:
        st.markdown("**Self-Exam**\n\nMonthly familiarity checks\n(Best 3-5 days after period)")

    # Treatment options with icons
    st.subheader("💊 Treatment Options")
    treat_col1, treat_col2, treat_col3, treat_col4 = st.columns(4)
    with treat_col1:
        st.markdown("**🔪 Surgery**\n\nLumpectomy/Mastectomy")
    with treat_col2:
        st.markdown("**☢️ Radiation**\n\nTargeted cell destruction")
    with treat_col3:
        st.markdown("**💉 Chemotherapy**\n\nSystemic drug treatment")
    with treat_col4:
        st.markdown("**💊 Hormone Therapy**\n\nBlocks estrogen/progesterone")

    # Visual footer
    st.markdown("---")
    st.success("**Early detection is key!** Talk to your doctor about personalized screening recommendations.")
        
if __name__ == '__main__':
    main()