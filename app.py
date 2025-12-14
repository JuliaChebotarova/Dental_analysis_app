import streamlit as st
from ultralytics import YOLO
from PIL import Image

import os
import gdown

# Налаштування сторінки
st.set_page_config(page_title="Dental Analysis", page_icon="🦷", layout="wide")

st.title("Діагностичний аналіз зубних знімків")

# Завантаження моделі
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        file_id = 'https://drive.google.com/file/d/1Fg-cp9PFqawFki7PM2fiGWhPOhY_ppeK/view?usp=sharing' 
        url = f'https://drive.google.com/uc?id=1Fg-cp9PFqawFki7PM2fiGWhPOhY_ppeK'
        gdown.download(url, "best.pt", quiet=False)
    model = YOLO("best.pt")
    return model

model = load_model()

# Бічна панель налаштувань
st.sidebar.header("Налаштування")
conf_threshold = st.sidebar.slider(
    "Яку точність результатів бажаєте?", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.70, 
    step=0.05,
    help="Чим вище значення, тим точніші результати, але може бути менше виявлених патологій"
)

# Завантаження фото
uploaded_file = st.file_uploader('Завантажте знімок і дізнайтеся про свої патології:', type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Оригінал")
        st.image(image, use_container_width=True)

    if st.sidebar.button("Дізнатися патології", type="primary"):
        with st.spinner('Триває аналіз...'):
            
# Новий результат
            results = model.predict(image, conf=conf_threshold)
            res = results[0]
            plotted_image = res.plot()
            with col2:
                st.header("Результат")
                st.image(plotted_image, channels="BGR", use_container_width=True)
            count = len(res.boxes)
            if count > 0:
                st.info(f"Знайдено патологій: {count}")
            else:

                st.warning("Патологій не знайдено.")



