# HealthCare Chatbot

Healthcare chatbot system

Developed using **FastAPI** for Back-end and **Streamlit** for Front-end

Use Guardrails to create gard, forbidden words to answer, and LLM use Typhoon

---

## How to install and use the project

1. **Create a Virtual Environment**

- **For Windows**

Open Command Prompt and run the command
```
py -3.12 -m venv venv
```
*Note: Use python ver 3.12.x, but don't worry, just run it in the folder of venv that supports this version
*Can still use version 3.13.x as usual, just run in the venv folder
- **For macOS / Linux**
Open Terminal and run the command
```
python3 -m venv venv
```
and create a .env file with
``` TYPHOON_API_KEY= your_key```
Get the api_key from https://playground.opentyphoon.ai/api-key
3. **Install the required libraries**
- Go to the Virtual Environment (activate venv) first
- **Windows:**
```
venv\Scripts\activate
```
- **macOS / Linux:**
```
source venv/bin/activate
```
- Then install dependencies with the command
```
pip install -r requirements.txt
```

4. **Activate the Virtual Environment (Every time before running the program)**
- **Windows:**
```
venv\Scripts\activate
```
- **macOS / Linux:**
```
source venv/bin/activate
```

5. **Run Backend (FastAPI)**
- Use the command
```
uvicorn main:app --reload
```
- The API will be available at [http://localhost:8000](http://localhost:8000)

6. **Run Frontend (Streamlit)**
- If you want to test with the UI page, use the command
```
streamlit run app_streamlit.py
```
- The UI will be accessible at [http://localhost:8501](http://localhost:8501)

- If you want to test at CLI
```
python app.py
```

---

> **Note**
> - Every time you run backend or frontend, you must activate venv first.
> - It is recommended to add `venv/` to the `.gitignore` file to prevent pushing venv to GitHub.
> - Don't forget to upload the custom_cnn_dfu_model.h5 file because it is used as data for prediction.
