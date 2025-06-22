# HealthCare Chatbot

ระบบแชทบอทสำหรับดูแลสุขภาพ  
พัฒนาโดยใช้ **FastAPI** สำหรับ Back-end และ **Streamlit** สำหรับ Front-end

ใช้ Guardrails สำหรับสร้าง gard คำต้องห้ามตอบ และ  LLM ใช้ Typhoon

---

## วิธีติดตั้งและใช้งานโปรเจกต์

1. **สร้าง Virtual Environment**

   - **สำหรับ Windows**  
     เปิด Command Prompt แล้วรันคำสั่ง  
     ```
     py -3.12 -m venv venv
     ```

   - **สำหรับ macOS / Linux**  
     เปิด Terminal แล้วรันคำสั่ง  
     ```
     python3 -m venv venv
     ```
   และสร้างไฟล์ .env พร้อมใส่
  ```  TYPHOON_API_KEY= your_key```
  รับ api_key มาจาก https://playground.opentyphoon.ai/api-key
3. **ติดตั้งไลบรารีที่จำเป็น**  
   - ให้เข้าไปที่ Virtual Environment (activate venv) ก่อน  
     - **Windows:**  
       ```
       venv\Scripts\activate
       ```
     - **macOS / Linux:**  
       ```
       source venv/bin/activate
       ```
   - จากนั้นติดตั้ง dependencies ด้วยคำสั่ง  
     ```
     pip install -r requirements.txt
     ```

4. **เปิดใช้งาน Virtual Environment (ทุกครั้งก่อนรันโปรแกรม)**  
   - **Windows:**  
     ```
     venv\Scripts\activate
     ```
   - **macOS / Linux:**  
     ```
     source venv/bin/activate
     ```

5. **รัน Backend (FastAPI)**  
   - ใช้คำสั่ง  
     ```
     uvicorn main:app --reload
     ```
   - API จะพร้อมใช้งานที่ [http://localhost:8000](http://localhost:8000)

6. **รัน Frontend (Streamlit)**  
   - ใช้คำสั่ง  
     ```
     streamlit run app.py
     ```
   - UI จะสามารถเข้าใช้งานได้ที่ [http://localhost:8501](http://localhost:8501)

---

> **หมายเหตุ**  
> - ทุกครั้งที่รัน backend หรือ frontend ต้อง activate venv ก่อนเสมอ  
> - แนะนำให้เพิ่ม `venv/` ลงในไฟล์ `.gitignore` เพื่อป้องกันไม่ให้ push venv ขึ้น GitHub

