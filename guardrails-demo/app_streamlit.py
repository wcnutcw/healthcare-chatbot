import streamlit as st
from guardrails import Guard
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import json
from predict import (
    load_symptom_data,
    extract_symptoms_from_text,
    predict_disease_percent
)
from health_prompt_template import (
    get_ai1_consistency_template,
    get_ai2_summary_template,
    get_ai3_doctor_reply_template,
    get_skin_image_summary_template,
)
from skin_model_predict import predict_skin_disease
import warnings
from PIL import Image

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

# ===== Guardrails หลายไฟล์ สำหรับแต่ละ AI
guard_ai1 = Guard.from_rail("guardrails_spec_ai1.rail")
guard_ai2 = Guard.from_rail("guardrails_spec_ai2.rail")
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

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def convert_json_to_str(json_data):
    # ใช้ json.dumps() เพื่อแปลงข้อมูลทุกอย่างใน JSON เป็น string
    return json.dumps(json_data, ensure_ascii=False)

def format_ai3_bullet(text):
    lines = text.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith('•'):
            if i > 0 and lines[i-1].strip() != '':
                new_lines.append('')  # เพิ่มบรรทัดว่างระหว่าง bullet
        new_lines.append(line)
    return '\n'.join(new_lines)

# =========================
def typhoon_wrapper(prompt, **kwargs):
    model = kwargs.get("model", "typhoon-v2.1-12b-instruct")
    temperature = kwargs.get("temperature", 0.3)
    max_tokens = kwargs.get("max_new_tokens", 512)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "คุณเป็นผู้ช่วย AI สุขภาพเบื้องต้น พูดจาอ่อนโยน ให้ข้อมูลเหมือนผู้หญิงไทย สุภาพ เป็นมิตร ไม่พูด 'สวัสดี' ทุกครั้ง (พูดแค่ทักทายครั้งแรกเท่านั้น) และห้ามวินิจฉัยหรือสั่งยา ต้องแนะนำให้พบแพทย์เสมอ"},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content

# ================= AI 3 CHAIN =================
def ai_chain_consistency(user_symptoms, predicted_diseases, llm_api, json_file):
    json_data = json_file
    disease_info = json_data
    predicted_diseases_str = "\n".join([f"{i+1}. {d} {p}% (จาก {m} อาการ)" for i, (d, p, m) in enumerate(predicted_diseases)])
    prompt_template = get_ai1_consistency_template()
    prompt = prompt_template.format(
        user_symptoms=", ".join(user_symptoms),
        predicted_diseases=predicted_diseases_str,
        json_data=disease_info
    )
    response = guard_ai1(
        prompt=prompt,
        llm_api=llm_api,
        llm_params={"model": "typhoon-v2.1-12b-instruct", "temperature": 0.2, "max_new_tokens": 256}
    )
    return response.validated_output if response.validated_output else {}

def ai_chain_summary(user_symptoms, predicted_diseases, ai1_comment, llm_api):
    prompt_template = get_ai2_summary_template()
    prompt = prompt_template.format(
        user_symptoms=", ".join(user_symptoms),
        predicted_diseases="\n".join([f"{i+1}. {d} {p}% (จาก {m} อาการ)" for i, (d, p, m) in enumerate(predicted_diseases)]),
        ai1_comment=ai1_comment or "-"
    )
    response = guard_ai2(
        prompt=prompt,
        llm_api=llm_api,
        llm_params={"model": "typhoon-v2.1-12b-instruct", "temperature": 0.2, "max_new_tokens": 512}
    )
    return response.validated_output if response.validated_output else {}

def ai_chain_doctor_reply(ai2_summary, ai2_recommendation, llm_api):
    prompt_template = get_ai3_doctor_reply_template()
    prompt = prompt_template.format(
        ai2_summary=ai2_summary or "-",
        ai2_recommendation=ai2_recommendation or "-"
    )
    response = llm_api(prompt, model="typhoon-v2.1-12b-instruct", temperature=0.2, max_new_tokens=512)
    return response

# ================= NEW: AI CHAIN FOR SKIN DISEASE =================
def ai_chain_skin_summary(image_class, confidence, llm_api):
    """สร้างสรุปและคำแนะนำเบื้องต้นสำหรับการวิเคราะห์ภาพผิวหนัง"""
    if image_class == "Abnormal(Ulcer)":
        ai2_summary = f"จากการวิเคราะห์ภาพ พบลักษณะผิดปกติที่อาจเป็นแผลหรือรอยโรคผิวหนัง (ความมั่นใจ {confidence:.1%})"
        ai2_recommendation = "ควรปรึกษาแพทย์ผิวหนังเพื่อรับการตรวจและรักษาที่เหมาะสม"
    else:  # Normal(Healthy skin)
        ai2_summary = f"จากการวิเคราะห์ภาพ ผิวหนังดูปกติ (ความมั่นใจ {confidence:.1%})"
        ai2_recommendation = "ควรดูแลรักษาความสะอาดและความชุ่มชื้นของผิวหนังต่อไป"
    
    return ai2_summary, ai2_recommendation

def ai_chain_skin_doctor_reply(image_class, confidence, llm_api):
    """สร้างคำตอบจากหมอสำหรับการวิเคราะห์ภาพผิวหนัง"""
    ai2_summary, ai2_recommendation = ai_chain_skin_summary(image_class, confidence, llm_api)
    
    prompt_template = get_skin_image_summary_template()
    prompt = prompt_template.format(
        image_class=f"{image_class} (ความมั่นใจ {confidence:.1%})",
        ai2_summary=ai2_summary,
        ai2_recommendation=ai2_recommendation
    )
    
    response = llm_api(prompt, model="typhoon-v2.1-12b-instruct", temperature=0.2, max_new_tokens=512)
    return response

# =========================
# ฟังก์ชันการถามบอท
def ask_bot_streamlit(user_message, n_results=1, greeted=False):
    msg_lower = user_message.lower().strip()

    if any(word in msg_lower for word in THANK_WORDS):
        return random.choice(THANK_REPLIES)

    if any(word in msg_lower for word in HOW_ARE_YOU_WORDS):
        return random.choice(HOW_ARE_YOU_REPLIES)

    for disease in known_diseases:
        if disease in user_message or disease in msg_lower:
            prompt = f"ผู้ใช้แจ้งว่าตนเองอาจเป็น '{disease}'. กรุณาให้คำแนะนำเบื้องต้นเกี่ยวกับโรคนี้ (โดยไม่วินิจฉัย ไม่สั่งยา) และเน้นให้พบแพทย์หากไม่แน่ใจอาการ"
            response = guard(
                prompt=prompt,
                llm_api=typhoon_wrapper,
                llm_params={"model": "typhoon-v2.1-12b-instruct", "temperature": 0.3, "max_new_tokens": 512}
            )
            if response.validated_output and isinstance(response.validated_output, dict):
                answer = response.validated_output.get("answer")
                if answer:
                    return answer.strip()
            return "ขออภัยค่ะ ดิฉันไม่สามารถให้ข้อมูลได้ในขณะนี้ หากมีอาการผิดปกติควรปรึกษาแพทย์นะคะ"

    if not greeted and any(word in msg_lower for word in GENERAL_GREET_WORDS):
        return random.choice(GENERAL_GREET_REPLIES)

    if "ยา" in msg_lower or "แนะนำยา" in msg_lower:
        return "ขออภัยค่ะ ดิฉันไม่สามารถแนะนำหรือสั่งยาได้ หากมีอาการผิดปกติควรปรึกษาเภสัชกรหรือแพทย์โดยตรงนะคะ"

    matched_symptoms = extract_symptoms_from_text(user_message, known_symptoms)
    if not matched_symptoms:
        return "ขออภัยค่ะ ดิฉันไม่เข้าใจอาการที่ระบุ กรุณาพิมพ์อาการให้ชัดเจน เช่น ปวดหัว มีไข้ ไอ หรืออื่นๆ"

    results = predict_disease_percent(matched_symptoms, df, disease_col)
    n_show = 3 if n_results < 1 else n_results
    results = results[:n_show]

    json_file_path = './symptoms_data.json'
    json_data = load_json_file(json_file_path)
    json_data_str = convert_json_to_str(json_data)
    ai1_res = ai_chain_consistency(matched_symptoms, results, typhoon_wrapper, json_data_str)
    ai1_comment = ai1_res.get('comment', '')

    ai2_res = ai_chain_summary(matched_symptoms, results, ai1_comment, typhoon_wrapper)
    ai2_summary = ai2_res.get('summary', '')
    ai2_recommendation = ai2_res.get('recommendation', '')

    ai3_reply = ai_chain_doctor_reply(ai2_summary, ai2_recommendation, typhoon_wrapper)
    ai3_reply = format_ai3_bullet(ai3_reply)

    st.session_state.ai1_res = ai1_res
    st.session_state.ai2_res = ai2_res
    st.session_state.ai3_reply = ai3_reply

    return ai3_reply.strip()

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="AI Health Symptom Advisor", page_icon="💊")

st.markdown("""
<style>
.messenger-container {max-width:700px; margin:0 auto;}
.messenger-bubble-row {display: flex; margin-bottom: 18px;}
.messenger-bubble {
    padding: 10px 18px;
    border-radius: 20px;
    font-size: 1.10rem;
    max-width: 72%;
    word-break: break-word;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    min-width: 60px;
    display:inline-block;
}
.messenger-bubble-user {
    background: #3b7ddd;
    color: #fff;
    margin-left: auto;
    margin-right: 0;
    border-bottom-right-radius: 8px;
    text-align: right;
}
.messenger-bubble-ai {
    background: #e6eaf1;
    color: #222;
    margin-right: auto;
    margin-left: 0;
    border-bottom-left-radius: 8px;
    text-align: left;
}
@media (prefers-color-scheme: dark) {
    .messenger-bubble-ai {background: #232632; color: #eee;}
    .messenger-bubble-user {background: #397cf8; color: #fff;}
}
</style>
""", unsafe_allow_html=True)

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

# **เพิ่มตัวแปรเก็บผลวิเคราะห์ภาพ**
if "ai3_skin_reply" not in st.session_state:
    st.session_state.ai3_skin_reply = ""
if "skin_analysis_result" not in st.session_state:
    st.session_state.skin_analysis_result = None

# ----------------- Messenger Bubble Layout ----------------
st.markdown('<div class="messenger-bg">', unsafe_allow_html=True)
st.markdown('<div class="messenger-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="messenger-bubble-row" style="justify-content:flex-end;">'
            f'  <div class="messenger-bubble messenger-bubble-user">{msg["content"]}</div>'
            f'</div>', unsafe_allow_html=True)
    elif msg["role"] == "ai":
        st.markdown(
            f'<div class="messenger-bubble-row" style="justify-content:flex-start;">'
            f'  <div class="messenger-bubble messenger-bubble-ai">{msg["content"]}</div>'
            f'</div>', unsafe_allow_html=True)

if st.session_state.pending_ai:
    st.markdown(
        '<div class="messenger-bubble-row" style="justify-content:flex-start;">'
        '<div class="messenger-bubble messenger-bubble-ai">กำลังพิมพ์...</div>'
        '</div>', unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True) # .messenger-container
st.markdown('</div>', unsafe_allow_html=True) # .messenger-bg

# --- เพิ่ม UI อัปโหลดรูปภาพสำหรับวิเคราะห์ ---
st.sidebar.title("🔬 วิเคราะห์โรคผิวหนังจากรูปภาพ")
st.sidebar.markdown("อัปโหลดภาพผิวหนังเพื่อให้ AI วิเคราะห์เบื้องต้น")

uploaded_file = st.sidebar.file_uploader("เลือกรูปภาพผิวหนัง", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.sidebar.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    if st.sidebar.button("🔍 วิเคราะห์ภาพ", type="primary"):
        with st.spinner("กำลังวิเคราะห์ภาพ..."):
            try:
                predicted_class, confidence = predict_skin_disease(image)
                
                # สร้างคำตอบจาก AI Doctor
                skin_ai3_reply = ai_chain_skin_doctor_reply(predicted_class, confidence, typhoon_wrapper)
                skin_ai3_reply = format_ai3_bullet(skin_ai3_reply)
                
                # เก็บผลลัพธ์ใน session state
                st.session_state.ai3_skin_reply = skin_ai3_reply
                st.session_state.skin_analysis_result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "reply": skin_ai3_reply
                }
                
                st.sidebar.success("✅ วิเคราะห์เสร็จแล้ว!")
                
            except Exception as e:
                st.sidebar.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

# แสดงผลการวิเคราะห์ภาพ
if st.session_state.skin_analysis_result:
    st.sidebar.markdown("### 📋 ผลการวิเคราะห์")
    result = st.session_state.skin_analysis_result
    
    # แสดงผลการจำแนกประเภท
    if result["predicted_class"] == "Abnormal(Ulcer)":
        st.sidebar.warning(f"⚠️ **พบความผิดปกติ**")
    else:
        st.sidebar.success(f"✅ **ผิวหนังปกติ**")
    
    st.sidebar.info(f"ความมั่นใจ: {result['confidence']:.1%}")
    
    # แสดงคำแนะนำจากหมอ
    st.sidebar.markdown("### 💬 คำแนะนำจากแพทย์ AI")
    st.sidebar.markdown(result["reply"])

# แชทปกติ
user_input = st.chat_input("พิมพ์ข้อความของคุณที่นี่")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_ai = True
    st.rerun()

if st.session_state.pending_ai:
    user_message = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"][-1]
    bot_reply = ask_bot_streamlit(user_message, n_results=1, greeted=st.session_state.greeted)
    st.session_state.messages.append({"role": "ai", "content": bot_reply})

    if not st.session_state.greeted:
        st.session_state.greeted = True
    st.session_state.pending_ai = False
    st.rerun()

# --- DEBUG --- (แสดงผลใน sidebar แยกจากวิเคราะห์ผิวหนัง)
with st.sidebar.expander("🛠️ DEBUG - รายละเอียดการประมวลผล", expanded=False):
    if "ai1_res" in st.session_state:
        st.markdown("🟦 **AI1 (Consistency Check)**")
        st.json(st.session_state.ai1_res)
    
    if "ai2_res" in st.session_state:
        st.markdown("🟩 **AI2 (Summary & Recommend)**")
        st.json(st.session_state.ai2_res)
    
    if "ai3_reply" in st.session_state:
        st.markdown("🟧 **AI3 (Doctor Reply - Text)**")
        st.write(st.session_state.ai3_reply)

    if "ai3_skin_reply" in st.session_state and st.session_state.ai3_skin_reply:
        st.markdown("🟪 **AI3 (Doctor Reply - Skin Image Analysis)**")
        st.write(st.session_state.ai3_skin_reply)