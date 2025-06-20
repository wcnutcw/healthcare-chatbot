# predict.py
import pandas as pd
from rapidfuzz import process
from collections import defaultdict

def load_symptom_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # ค้นหาคอลัมน์โรคที่แท้จริง
    disease_col = None
    for d in ["diagnosis", "disease", "โรค"]:
        if d in df.columns:
            disease_col = d
            break
    if disease_col is None:
        raise ValueError("ไม่พบคอลัมน์ diagnosis/disease/โรค ในไฟล์ CSV")
    # อาการทั้งหมดคือทุกคอลัมน์ที่ไม่ใช่โรค
    known_symptoms = [col for col in df.columns if col != disease_col and not col.startswith("Unnamed")]
    return df, known_symptoms, disease_col

def extract_symptoms_from_text(user_text, known_symptoms, threshold=80):
    words = user_text.replace('และ', ' ').replace(',', ' ').split()
    matched = set()
    for word in words:
        res = process.extractOne(word, known_symptoms, score_cutoff=threshold)
        if res is not None:
            match = res[0]
            matched.add(match)
    return list(matched)

def predict_disease_percent(symptom_list, df, disease_col):
    summary = defaultdict(lambda: {"total_match": 0, "case_count": 0, "max_symptom": 0})
    for _, row in df.iterrows():
        disease = row[disease_col]
        matched = sum([row[symptom] for symptom in symptom_list if symptom in df.columns])
        max_total = len(symptom_list)
        if max_total > 0:
            summary[disease]["total_match"] += matched / max_total
            summary[disease]["case_count"] += 1
            summary[disease]["max_symptom"] = max_total
    results = []
    for disease, stats in summary.items():
        avg_percent = round((stats["total_match"] / stats["case_count"]) * 100, 2) if stats["case_count"] > 0 else 0.0
        results.append((disease, avg_percent, stats["max_symptom"]))
    # เรียงจากโรคที่ตรงกับอาการมากที่สุด
    return sorted(results, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    csv_path = '/guardrails-demo/data/full_onehot_disease.csv'
    df, known_symptoms, disease_col = load_symptom_data(csv_path)
    user_text = input("โปรดพิมพ์อาการของคุณเป็นประโยค: ")
    matched_symptoms = extract_symptoms_from_text(user_text, known_symptoms)
    print("\nอาการที่ระบบเข้าใจ:", matched_symptoms)
    results = predict_disease_percent(matched_symptoms, df, disease_col)
    print("\nระบบวิเคราะห์ว่าอาจเป็นโรคต่อไปนี้:")
    for i, (disease, percent, max_symptom) in enumerate(results[:5]):
        print(f"{i+1}. {disease}: {percent}% (จากอาการทั้งหมด {max_symptom})")
