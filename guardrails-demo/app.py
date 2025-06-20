# app.py
from guardrails import Guard
from openai import OpenAI
from dotenv import load_dotenv
import os

from predict import (
    load_symptom_data,
    extract_symptoms_from_text,
    predict_disease_percent
)
from health_prompt_template import get_health_prompt_template

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

guard = Guard.from_rail("guardrails_spec.rail")

def ask_bot(user_message, n_results=1, greeted=False):
    # ทักทายครั้งแรก
    greet_words = {"สวัสดี", "hello", "hi", "ดีครับ", "ดีค่ะ"}
    msg_lower = user_message.lower().strip()

    if not greeted and any(word in msg_lower for word in greet_words):
        return (
            "สวัสดีค่ะ ดิฉันเป็นผู้ช่วย AI ด้านสุขภาพเบื้องต้น สามารถพิมพ์อาการหรือสอบถามข้อมูลสุขภาพได้เลยนะคะ "
            "แต่ดิฉันจะไม่วินิจฉัยหรือสั่งยา หากมีอาการผิดปกติควรพบแพทย์ค่ะ"
        )

    if "ยา" in msg_lower or "แนะนำยา" in msg_lower:
        return (
            "ขออภัยค่ะ ดิฉันไม่สามารถแนะนำหรือสั่งยาได้ "
            "หากมีอาการผิดปกติควรปรึกษาเภสัชกรหรือแพทย์โดยตรงนะคะ"
        )

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
            # ตัดสวัสดีที่ AI ตอบมาเอง (ตัดประโยคต้นทางถ้ามี "สวัสดี" หรือ "สวัสดีค่ะ" หรือ "สวัสดีครับ")
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

if __name__ == "__main__":
    print("=== AI Health Symptom Advisor (พิมพ์ exit เพื่อออก) ===")
    greeted = False
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not greeted and user_input.strip():
            bot_reply = ask_bot(user_input, n_results=1, greeted=False)
            greeted = True
        else:
            bot_reply = ask_bot(user_input, n_results=1, greeted=True)
        print("Bot:", bot_reply)
