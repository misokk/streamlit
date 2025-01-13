import streamlit as st
import matplotlib.pyplot as plt
from rmn import RMN
import cv2
import numpy as np
from PIL import Image

st.title("감정 분석")
st.write("이미지")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

m = RMN()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.write("분석중")
    results = m.detect_emotion_for_single_frame(img_bgr)

    st.write("Raw Results:", results)

    if results:
        st.success("분석 완료")

        emotion_percentages = {}
        for result in results:
            proba_list = result.get("proba_list", [])
            for emotion_data in proba_list:
                for emotion, proba in emotion_data.items():
                    if emotion in emotion_percentages:
                        emotion_percentages[emotion] += proba
                    else:
                        emotion_percentages[emotion] = proba

        total_proba = sum(emotion_percentages.values())
        if total_proba > 0:
            emotion_percentages = {emotion: (proba / total_proba * 100) for emotion, proba in emotion_percentages.items()}

            st.write("감정 확률 기반 비율:")
            for emotion, percent in emotion_percentages.items():
                st.write(f"{emotion}: {percent:.2f}%")

            st.write("감정 확률 기반 분포 그래프:")
            plt.figure(figsize=(8, 4))
            plt.bar(emotion_percentages.keys(), emotion_percentages.values(), color='skyblue')
            plt.xlabel("Emotion")
            plt.ylabel("Percentage (%)")
            plt.title("Emotion Distribution Based on Probabilities")
            plt.xticks(rotation=45)
            st.pyplot(plt)
        else:
            st.warning("감정 확률 합계가 0입니다. 결과를 확인하세요.")

        img_with_results = m.draw(img_bgr, results)
        img_with_results = cv2.cvtColor(img_with_results, cv2.COLOR_BGR2RGB)
        st.image(img_with_results, caption="Detected Emotions", use_column_width=True)
    else:
        st.warning("사람 얼굴 아님")
