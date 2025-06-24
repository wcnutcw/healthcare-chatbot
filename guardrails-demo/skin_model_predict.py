from keras.models import load_model
import numpy as np
from PIL import Image
import streamlit as st

MODEL_PATH = './custom_cnn_dfu_model.h5'
CLASS_NAMES = ['Abnormal(Ulcer)', 'Normal(Healthy skin)']
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_skin_model():
    """โหลดโมเดล AI สำหรับวิเคราะห์ผิวหนัง"""
    try:
        # ✅ แก้ไขตรงนี้: ไม่ compile โมเดลเพื่อลด warning
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {str(e)}")
        return None

def predict_skin_disease(img_pil: Image.Image):
    """
    วิเคราะห์โรคผิวหนังจากภาพ

    Args:
        img_pil (Image.Image): ภาพ PIL ที่ต้องการวิเคราะห์

    Returns:
        Tuple[str, float]: (predicted_class, confidence)
    """
    model = load_skin_model()

    if model is None:
        raise Exception("ไม่สามารถโหลดโมเดลได้")

    try:
        # ปรับขนาดและ normalize
        img = img_pil.resize(IMAGE_SIZE)
        img_array = np.array(img).astype('float32') / 255.0

        # รองรับ grayscale และ alpha channel
        if img_array.ndim == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # เพิ่มมิติ batch
        img_array = np.expand_dims(img_array, axis=0)

        # ทำนาย
        prediction = model.predict(img_array, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return predicted_class, confidence

    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {str(e)}")


def get_skin_condition_description(predicted_class: str, confidence: float) -> str:
    """
    คืนค่าคำอธิบายของผลลัพธ์การวิเคราะห์ผิวหนังแบบภาษาไทย

    Args:
        predicted_class (str): คลาสที่โมเดลทำนายได้
        confidence (float): ความมั่นใจของโมเดล (0-1)

    Returns:
        str: คำตอบสรุปที่เหมาะสำหรับแสดงในหน้าเว็บ
    """
    if predicted_class == 'Abnormal(Ulcer)':
        return (
            f"🔍 ตรวจพบความผิดปกติที่อาจเป็นแผลเกิดจากโรคเบาหวาน "
            f"(ความมั่นใจ {confidence*100:.2f}%)\n\n"
            "📌 คำแนะนำ: ควรพบแพทย์เฉพาะทางหรือคลินิกโรคผิวหนังเพื่อวินิจฉัยเพิ่มเติมค่ะ"
        )
    elif predicted_class == 'Normal(Healthy skin)':
        return (
            f"✅ ไม่พบความผิดปกติจากภาพที่วิเคราะห์ "
            f"(ความมั่นใจ {confidence*100:.2f}%)\n\n"
            "📌 อย่างไรก็ตาม หากยังมีอาการผิดปกติ ควรปรึกษาแพทย์เพื่อความแน่ใจนะคะ"
        )
    else:
        return "ไม่สามารถประเมินผลได้จากภาพนี้ค่ะ กรุณาลองใหม่หรือลองใช้ภาพอื่น"
