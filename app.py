import streamlit as st
from ultralytics import YOLO
from PIL import Image

import os
import gdown

# --- НАЛАШТУВАННЯ СТОРІНКИ ---
st.set_page_config(page_title="Dental Analysis", page_icon="🦷", layout="wide")

st.title("Діагностичний аналіз зубних знімків")
st.write("Завантажте знімок і дізнайтеся про свої патології!")

# # --- 1. ЗАВАНТАЖЕННЯ МОДЕЛІ ---
# # Ми кешуємо модель, щоб вона не завантажувалась заново при кожному кліку
# @st.cache_resource
# def load_model():
#     # Важливо: файл best.pt має лежати поруч з app.py
#     # Якщо ваша модель називається інакше, змініть назву тут
#     model = YOLO("best.pt")
#     return model

# model = load_model()

@st.cache_resource
def load_model():
    # Перевіряємо, чи є файл локально
    if not os.path.exists("best.pt"):
        file_id = 'https://drive.google.com/file/d/1Fg-cp9PFqawFki7PM2fiGWhPOhY_ppeK/view?usp=sharing' 
        
        url = f'https://drive.google.com/uc?id=1Fg-cp9PFqawFki7PM2fiGWhPOhY_ppeK'
        # st.info("Завантажую модель з хмари... Це займе хвилинку ⏳")
        gdown.download(url, "best.pt", quiet=False)
        # st.success("Модель завантажено!")

    model = YOLO("best.pt")
    return model

# try:
#     model = load_model()
#     st.success("✅ Модель успішно завантажена!")
# except Exception as e:
#     st.error(f"❌ Не знайдено файл моделі 'best.pt'. Переконайтеся, що він у цій папці. Помилка: {e}")
#     st.stop()



# --- 2. БІЧНА ПАНЕЛЬ НАЛАШТУВАНЬ ---
st.sidebar.header("Налаштування")
# Поріг впевненості (Confidence Threshold)
conf_threshold = st.sidebar.slider(
    "Яку точність результатів бажаєте?", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.70, 
    step=0.05,
    help="Чим вище значення, тим точніші результати, але може бути менше виявлених патологій"
)

# --- 3. ЗАВАНТАЖЕННЯ ФОТО ---
uploaded_file = st.file_uploader("Оберіть зображення (JPG, PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Відкриваємо картинку
    image = Image.open(uploaded_file)

    # Створюємо колонки для порівняння
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Оригінал")
        st.image(image, use_container_width=True)

    # Кнопка запуску
    if st.sidebar.button("Дізнатися патології", type="primary"):
        with st.spinner('Триває аналіз...'):
            # --- 4. ПЕРЕДБАЧЕННЯ (INFERENCE) ---
            # Викликаємо модель YOLO прямо на картинці
            results = model.predict(image, conf=conf_threshold)

            # YOLO повертає список результатів, беремо перший (для однієї картинки)
            res = results[0]
            
            # Малюємо бокси/маски прямо на картинці
            # res.plot() повертає масив numpy (BGR), тому треба вказати канали
            plotted_image = res.plot()

            with col2:
                st.header("Результат")
                # channels="BGR" важливо, бо OpenCV (який всередині YOLO) використовує BGR
                st.image(plotted_image, channels="BGR", use_container_width=True)
                
            # Додаткова статистика (скільки об'єктів знайдено)
            count = len(res.boxes)
            if count > 0:
                st.info(f"Знайдено патологій: {count}")
            else:
                st.warning("Патологій не знайдено.")