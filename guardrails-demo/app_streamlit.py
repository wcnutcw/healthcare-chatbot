import streamlit as st
from guardrails import Guard
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from predict import (
    load_symptom_data,
    extract_symptoms_from_text,
    predict_disease_percent
)
from health_prompt_template import get_health_prompt_template
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# โหลด .env
load_dotenv()

TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
TYPHOON_API_URL = "https://api.opentyphoon.ai/v1"

client = OpenAI(
    api_key=TYPHOON_API_KEY,
    base_url=TYPHOON_API_URL
)

SYMPTOM_CSV = "./data/full_onehot_disease.csv"
df, known_symptoms, disease_col = load_symptom_data(SYMPTOM_CSV)
known_diseases = list(df[disease_col].unique())  # สำหรับตรวจชื่อโรค

guard = Guard.from_rail("guardrails_spec.rail")

# =========================
# กลุ่มคำสนทนาทั่วไป
# =========================
THANK_WORDS = {"ขอบคุณ", "ขอบคุณค่ะ", "ขอบคุณครับ", "thank you", "ขอบใจ", "ซาบซึ้ง"}
THANK_REPLIES = [
    "ยินดีค่ะ 😊 หากมีอะไรให้ช่วยเหลือเพิ่มเติม แจ้งได้เลยนะคะ",
    "ด้วยความยินดีนะคะ ดูแลสุขภาพด้วยค่ะ",
    "ขอบคุณเช่นกันค่ะ หากมีคำถามเกี่ยวกับสุขภาพหรืออยากพูดคุยเพิ่มเติม สามารถทักมาได้ตลอดนะคะ",
    "ขอบคุณที่พูดคุยกับดิฉันค่ะ ขอให้สุขภาพแข็งแรงนะคะ"
]

GENERAL_GREET_WORDS = {"สวัสดี", "hello", "hi", "ดีครับ", "ดีค่ะ"}
GENERAL_GREET_REPLIES = [
    "สวัสดีค่ะ ดิฉันเป็นผู้ช่วย AI ด้านสุขภาพเบื้องต้นของคุณ พร้อมให้คำแนะนำและดูแลสุขภาพคุณเสมอนะคะ หากมีอาการไม่สบายหรืออยากปรึกษาเรื่องสุขภาพ พิมพ์เข้ามาได้เลยค่ะ 💖",
    "สวัสดีค่ะ ดิฉันคือ AI ผู้ช่วยดูแลสุขภาพเบื้องต้นค่ะ หากต้องการข้อมูลเกี่ยวกับสุขภาพหรือมีอาการที่อยากสอบถาม สามารถพูดคุยกับดิฉันได้ตลอดเวลานะคะ 😊",
    "สวัสดีค่ะ ดิฉันเป็น AI ผู้ช่วยสุขภาพของคุณ พร้อมรับฟังและให้คำแนะนำสุขภาพเบื้องต้น หากมีข้อสงสัยหรืออยากพูดคุย สามารถสอบถามได้เลยค่ะ 💬",
    "สวัสดีค่ะ ดิฉันเป็น AI ผู้ช่วยด้านสุขภาพ หากต้องการคำแนะนำหรือมีอาการที่ต้องการพูดคุย สามารถพิมพ์มาถามดิฉันได้เสมอค่ะ ดูแลสุขภาพด้วยนะคะ"
]

HOW_ARE_YOU_WORDS = {"สบายดีไหม", "how are you", "เป็นยังไงบ้าง"}
HOW_ARE_YOU_REPLIES = [
    "ขอบคุณที่ถามค่ะ ดิฉันเป็น AI ที่พร้อมช่วยเหลือเรื่องสุขภาพเสมอนะคะ 😊",
    "ดิฉันสบายดีค่ะ และพร้อมดูแลสุขภาพของคุณเสมอค่ะ",
    "ขอบคุณที่ทักมาถามนะคะ มีอะไรอยากปรึกษาเกี่ยวกับสุขภาพไหมคะ"
]

# =========================
def typhoon_wrapper(prompt, **kwargs):
    model = kwargs.get("model", "typhoon-v2.1-12b-instruct")
    temperature = kwargs.get("temperature", 0.3)
    max_tokens = kwargs.get("max_new_tokens", 512)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":
                "คุณเป็นผู้ช่วย AI สุขภาพเบื้องต้น พูดจาอ่อนโยน ให้ข้อมูลเหมือนผู้หญิงไทย สุภาพ เป็นมิตร ไม่พูด 'สวัสดี' ทุกครั้ง (พูดแค่ทักทายครั้งแรกเท่านั้น) และห้ามวินิจฉัยหรือสั่งยา ต้องแนะนำให้พบแพทย์เสมอ"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content

def ask_bot_streamlit(user_message, n_results=1, greeted=False):
    msg_lower = user_message.lower().strip()

    # --------- General: ขอบคุณ ---------
    if any(word in msg_lower for word in THANK_WORDS):
        return random.choice(THANK_REPLIES)

    # --------- General: สบายดีมั้ย ---------
    if any(word in msg_lower for word in HOW_ARE_YOU_WORDS):
        return random.choice(HOW_ARE_YOU_REPLIES)

    # --------- กรณี user พิมพ์ชื่อโรค ---------
    for disease in known_diseases:
        if disease in user_message or disease in msg_lower:
            prompt = (
                f"ผู้ใช้แจ้งว่าตนเองอาจเป็น '{disease}'. "
                "กรุณาให้คำแนะนำเบื้องต้นเกี่ยวกับโรคนี้ (โดยไม่วินิจฉัย ไม่สั่งยา) และเน้นให้พบแพทย์หากไม่แน่ใจอาการ "
                'โปรดตอบเป็น JSON เช่น: {"answer": "ข้อความแนะนำเกี่ยวกับโรคนี้"}'
            )
            response = guard(
                prompt=prompt,
                llm_api=typhoon_wrapper,
                llm_params={"model": "typhoon-v2.1-12b-instruct", "temperature": 0.3, "max_new_tokens": 512}
            )
            if response.validated_output and isinstance(response.validated_output, dict):
                answer = response.validated_output.get("answer")
                if answer:
                    return answer.strip()
            return (
                "ขออภัยค่ะ ดิฉันไม่สามารถให้ข้อมูลได้ในขณะนี้ หากมีอาการผิดปกติควรปรึกษาแพทย์นะคะ"
            )

    # --------- ทักทายครั้งแรก ---------
    if not greeted and any(word in msg_lower for word in GENERAL_GREET_WORDS):
        return random.choice(GENERAL_GREET_REPLIES)

    # --------- ขอ/ถามเรื่องยา ---------
    if "ยา" in msg_lower or "แนะนำยา" in msg_lower:
        return (
            "ขออภัยค่ะ ดิฉันไม่สามารถแนะนำหรือสั่งยาได้ "
            "หากมีอาการผิดปกติควรปรึกษาเภสัชกรหรือแพทย์โดยตรงนะคะ"
        )

    # --------- วิเคราะห์อาการ ---------
    matched_symptoms = extract_symptoms_from_text(user_message, known_symptoms)
    if not matched_symptoms:
        return (
            "ขออภัยค่ะ ดิฉันไม่เข้าใจอาการที่ระบุ กรุณาพิมพ์อาการให้ชัดเจน เช่น ปวดหัว มีไข้ ไอ หรืออื่นๆ "
            "ถ้าอาการไม่ดีขึ้น ควรไปพบแพทย์นะคะ"
        )

    results = predict_disease_percent(matched_symptoms, df, disease_col)
    n_show = 1 if n_results < 1 else n_results
    results = results[:n_show]

    prompt_template = get_health_prompt_template()
    disease_rank_str = '\n'.join([
        f"{i+1}. {disease} {percent}% (จากอาการทั้งหมด {max_symptom})"
        for i, (disease, percent, max_symptom) in enumerate(results)
    ])
    prompt = prompt_template.format(
        symptoms=", ".join(matched_symptoms),
        disease_ranking=disease_rank_str
    )

    response = guard(
        prompt=prompt,
        llm_api=typhoon_wrapper,
        llm_params={"model": "typhoon-v2.1-12b-instruct", "temperature": 0.3, "max_new_tokens": 512}
    )
    if response.validated_output and isinstance(response.validated_output, dict):
        answer = response.validated_output.get("answer")
        if answer:
            lines = answer.strip().split("\n")
            if lines and ("สวัสดี" in lines[0]):
                return "\n".join(lines[1:]).strip()
            return answer.strip()
    if hasattr(response, "error") and response.error:
        return f"[ERROR] {response.error}"
    if hasattr(response, "raw_llm_output"):
        print("LLM raw output:", response.raw_llm_output)
    return (
        "ขออภัยค่ะ ดิฉันไม่สามารถตอบคำถามนี้ได้ หากคุณมีอาการผิดปกติควรปรึกษาแพทย์นะคะ"
    )

# ---------- Streamlit UI -----------
st.set_page_config(page_title="AI Health Symptom Advisor", page_icon="💊")
st.title("💬 AI Health Symptom Advisor")
st.markdown(
    "พิมพ์อาการหรือสอบถามข้อมูลสุขภาพเบื้องต้น (บอทจะตอบแบบผู้หญิง อ่อนโยน ไม่วินิจฉัย ไม่แนะนำยา)\n\n"
    "**หมายเหตุ:** ข้อมูลนี้เป็นเพียงคำแนะนำเบื้องต้น หากอาการไม่ดีขึ้นควรปรึกษาแพทย์"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "pending_ai" not in st.session_state:
    st.session_state.pending_ai = False

# แสดงประวัติแบบ chat bubble ซ้าย-ขวา
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("ai"):
            st.markdown(msg["content"])

# แสดง "..." bubble ถ้ากำลังรอ AI
if st.session_state.pending_ai:
    with st.chat_message("ai"):
        st.markdown("กำลังพิมพ์...")

user_input = st.chat_input("พิมพ์ข้อความของคุณที่นี่")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_ai = True
    st.rerun()  # refresh เพื่อให้ bubble "กำลังพิมพ์..." โผล่

# ถ้ามี pending_ai = True และ message ล่าสุดคือ user → เรียก AI
if st.session_state.pending_ai:
    user_message = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"][-1]
    bot_reply = ask_bot_streamlit(user_message, n_results=1, greeted=st.session_state.greeted)
    st.session_state.messages.append({"role": "ai", "content": bot_reply})

    if not st.session_state.greeted:
        st.session_state.greeted = True
    st.session_state.pending_ai = False
    st.rerun()
