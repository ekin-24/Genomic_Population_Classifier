import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Population Detect",
    layout="wide"
)

# 26 Classes
POP_MAP = {
    'JPT': {'iso': 'JPN', 'name': 'Japan'},
    'CHB': {'iso': 'CHN', 'name': 'China'},
    'CHS': {'iso': 'CHN', 'name': 'China'},
    'CDX': {'iso': 'CHN', 'name': 'China'},
    'KHV': {'iso': 'VNM', 'name': 'Vietnam'},
    'FIN': {'iso': 'FIN', 'name': 'Finland'},
    'GBR': {'iso': 'GBR', 'name': 'United Kingdom'},
    'CEU': {'iso': 'USA', 'name': 'United States'},
    'IBS': {'iso': 'ESP', 'name': 'Spain'},
    'TSI': {'iso': 'ITA', 'name': 'Italy'},
    'PEL': {'iso': 'PER', 'name': 'Peru'},
    'MXL': {'iso': 'MEX', 'name': 'Mexico'},
    'CLM': {'iso': 'COL', 'name': 'Colombia'},
    'PUR': {'iso': 'PRI', 'name': 'Puerto Rico'},
    'YRI': {'iso': 'NGA', 'name': 'Nigeria'},
    'LWK': {'iso': 'KEN', 'name': 'Kenya'},
    'GWD': {'iso': 'GMB', 'name': 'Gambia'},
    'MSL': {'iso': 'SLE', 'name': 'Sierra Leone'},
    'ESN': {'iso': 'NGA', 'name': 'Nigeria'},
    'ASW': {'iso': 'USA', 'name': 'United States'},
    'ACB': {'iso': 'BRB', 'name': 'Barbados'},
    'GIH': {'iso': 'IND', 'name': 'India'},
    'BEB': {'iso': 'BGD', 'name': 'Bangladesh'},
    'ITU': {'iso': 'IND', 'name': 'India'},
    'PJL': {'iso': 'PAK', 'name': 'Pakistan'},
    'STU': {'iso': 'LKA', 'name': 'Sri Lanka'}
}

@st.cache_resource
def load_assets():
    model = load_model('1000genomesProject.keras')
    scaler = joblib.load('genetics_scaler.pkl') # Z score scaler
    le = joblib.load('label_encoder.pkl') # Labels
    selector = joblib.load('genetics_selector.pkl') # Feature selector
    return model, scaler, le, selector

# Checks all files are complete
try:
    model, scaler, le, selector = load_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error: {e}")


def draw_choropleth_map(prob_df):
    # Create world map from model outputs
    prob_df['ISO_Code'] = prob_df['Country Code'].apply(lambda x: POP_MAP[x]['iso'])
    prob_df['Country Name'] = prob_df['Country Code'].apply(lambda x: POP_MAP[x]['name'])

    # Combine different subgroups belonging to the same country
    country_prob = prob_df.groupby(['ISO_Code', 'Country Name'])['Possibility (%)'].sum().reset_index()

    # Plotly 3D Sphere Map
    fig = px.choropleth(
        country_prob,
        locations="ISO_Code",
        color="Possibility (%)",
        hover_name="Country Name",
        color_continuous_scale="RdYlGn",    
        range_color=[0, 100],               
        projection="orthographic",          
        labels={'Possibility (%)': 'Genetic Link'} 
    )

    # National borders
    fig.update_traces(marker_line_color="Black", marker_line_width=0.5)
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        geo=dict(
            showframe=True,
            framecolor='#333333',       
            showcoastlines=True,
            coastlinecolor="#333333",
            showland=True,
            landcolor="#2b2b2b",        
            showocean=True,
            oceancolor="#16161a",       
            showcountries=True,
            countrycolor="#444444",
            bgcolor='rgba(0,0,0,0)'
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar=dict(
            title=dict(text="Genetic Link", font=dict(color="white")),
            ticksuffix=" %",
            tickfont=dict(color="white"),
            bgcolor='rgba(0,0,0,0.5)'
        )
    )
    
    fig.update_geos(projection_rotation=dict(lon=45, lat=35, roll=0))
    st.plotly_chart(fig, use_container_width=True)


st.title("Genetic Population Classifier")

if model_loaded:
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload your DNA file (.npy):", type=["npy"])
    
    if uploaded_file is not None:
        if st.button("Analyze the DNA.", use_container_width=True):
            with st.spinner("Please wait..."):
                
                sample_data = np.load(uploaded_file).reshape(1, -1)
                scaled_input = scaler.transform(sample_data)
                cnn_input = np.expand_dims(scaled_input, axis=2)
                
                preds = model.predict(cnn_input)
                pred_idx = np.argmax(preds[0])
                confidence = float(np.max(preds[0]) * 100)
                predicted_pop_code = le.inverse_transform([pred_idx])[0]
                predicted_country = POP_MAP[predicted_pop_code]['name']
                
                prob_df = pd.DataFrame({'Country Code': le.classes_, 'Possibility (%)': preds[0] * 100})
                prob_df = prob_df.sort_values(by='Possibility (%)', ascending=False).reset_index(drop=True)
                
            st.success("The genetic sequence was successfully deciphered.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Estimated Country", value=predicted_country)
            with col2:
                st.metric(label="Model Confidence Score", value=f"%{confidence:.2f}")
                
            st.subheader("Distribution Map")
            draw_choropleth_map(prob_df) 
            
            with st.expander("View the Full Probability Distribution of Subpopulations"):
                st.dataframe(prob_df.style.format({'Possibility (%)': "{:.4f}%"}).background_gradient(cmap='RdYlGn', subset=['Possibility (%)']), use_container_width=True)

            report_text = f"========================================\nPopulation Analysis Report\n========================================\n\n Estimated Country : {predicted_country} ({predicted_pop_code})\n Model Confidence Score  : %{confidence:.2f}\n\n[ The 5 Most Likely Possibilities ]\n"
            for index, row in prob_df.head(5).iterrows():
                report_text += f"- {row['Country Code']:<5} : %{row['Possibility (%)']:.4f}\n"

            report_text += "\n Report."

            st.write("") 
            st.download_button(
                label="Download Analysis Result (.txt)", 
                data=report_text, 
                file_name="genetic_report.txt", 
                mime="text/plain", 
                use_container_width=True
            )
else:
    st.info("Error!")