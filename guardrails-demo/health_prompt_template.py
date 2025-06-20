# health_prompt_template.py
from langchain.prompts import PromptTemplate

def get_health_prompt_template():
    return PromptTemplate(
        input_variables=["symptoms", "disease_ranking"],
        template=(
            "อาการที่ผู้ใช้แจ้ง: {symptoms}\n"
            "ระบบวิเคราะห์ว่าอาจเป็นโรคต่อไปนี้ (เรียงตามเปอร์เซ็นต์):\n{disease_ranking}\n"
            "ขอคำแนะนำสำหรับอาการนี้ (ห้ามวินิจฉัย/ห้ามสั่งยา):\n"
            'โปรดตอบเป็น JSON เช่น: {{"answer": "ข้อความแนะนำเบื้องต้นและแจ้งให้พบแพทย์"}}'
        )
    )
