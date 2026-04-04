"""
error_analysis.py
=================
Phân tích lỗi cho mô hình Sentiment Analysis (best_sentiment_model + optimized_tfidf_sentiment)
Cấu trúc thư mục kỳ vọng:
  tuned_models/
    best_sentiment_model.pkl
    optimized_tfidf_sentiment.pkl
    tuning_info.pkl          (tuỳ chọn)
  models/
    sentiment_model.pkl      (baseline để so sánh)
    tfidf_sentiment.pkl
  vietnamese_students_feedback/
    test/
      sentences.txt
      sentiments.txt
"""

import pickle
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ─────────────────────────────────────────
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ─────────────────────────────────────────
TUNED_MODEL_PATH  = Path("tuned_models/best_sentiment_model.pkl")
TUNED_TFIDF_PATH  = Path("tuned_models/optimized_tfidf_sentiment.pkl")
BASE_MODEL_PATH   = Path("models/sentiment_model.pkl")
BASE_TFIDF_PATH   = Path("models/tfidf_sentiment.pkl")
TUNING_INFO_PATH  = Path("tuned_models/tuning_info.pkl")

TEST_SENT_PATH    = Path("vietnamese_students_feedback/test/sentences.txt")
TEST_LABEL_PATH   = Path("vietnamese_students_feedback/test/sentiments.txt")

OUTPUT_CSV        = Path("error_analysis_results.csv")

# ─────────────────────────────────────────
# 2. HÀM TIỀN XỬ LÝ (giống notebook)
# ─────────────────────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s\u0080-\u024F\u1E00-\u1EFF.,!?]", " ", text)
    return " ".join(text.split())

# ─────────────────────────────────────────
# 3. TẢI MÔ HÌNH
# ─────────────────────────────────────────
def load_model(model_path: Path, tfidf_path: Path, name: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    print(f"✓ Đã tải {name}: {type(model).__name__}")
    return model, tfidf

print("=" * 70)
print("🔍 PHÂN TÍCH LỖI - SENTIMENT MODEL")
print("=" * 70)

tuned_model, tuned_tfidf = load_model(TUNED_MODEL_PATH, TUNED_TFIDF_PATH, "Tuned Model")

has_baseline = BASE_MODEL_PATH.exists() and BASE_TFIDF_PATH.exists()
if has_baseline:
    base_model, base_tfidf = load_model(BASE_MODEL_PATH, BASE_TFIDF_PATH, "Baseline Model")

# ─────────────────────────────────────────
# 4. TẢI DỮ LIỆU TEST
# ─────────────────────────────────────────

# Mapping số → string (theo notebook gốc)
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive",
                 "0": "negative", "1": "neutral", "2": "positive"}

def normalize_label(lbl):
    """Chuẩn hoá label về string, dù file lưu số hay chữ."""
    return SENTIMENT_MAP.get(lbl, SENTIMENT_MAP.get(str(lbl), str(lbl)))

sentences = TEST_SENT_PATH.read_text(encoding="utf-8").strip().splitlines()
raw_labels = TEST_LABEL_PATH.read_text(encoding="utf-8").strip().splitlines()

df = pd.DataFrame({"sentence": sentences, "true_label": raw_labels})
# Chuẩn hoá true_label về string
df["true_label"] = df["true_label"].apply(normalize_label)
df["text_processed"] = df["sentence"].apply(preprocess)
df["word_count"]     = df["text_processed"].str.split().str.len()
print(f"\n📂 Dữ liệu test: {len(df)} mẫu")
print(f"   Phân phối nhãn:\n{df['true_label'].value_counts().to_string()}\n")

# ─────────────────────────────────────────
# 5. DỰ ĐOÁN
# ─────────────────────────────────────────
X_test = tuned_tfidf.transform(df["text_processed"])
raw_pred = tuned_model.predict(X_test)
df["tuned_pred"]    = [normalize_label(p) for p in raw_pred]
df["tuned_correct"] = df["tuned_pred"] == df["true_label"]

if has_baseline:
    X_test_base = base_tfidf.transform(df["text_processed"])
    raw_base_pred = base_model.predict(X_test_base)
    df["base_pred"]    = [normalize_label(p) for p in raw_base_pred]
    df["base_correct"] = df["base_pred"] == df["true_label"]

# ─────────────────────────────────────────
# 6. BÁO CÁO TỔNG QUAN
# ─────────────────────────────────────────
print("=" * 70)
print("📊 KẾT QUẢ TUNED MODEL")
print("=" * 70)
print(f"\nAccuracy : {accuracy_score(df['true_label'], df['tuned_pred']):.4f}")
print(f"F1 (weighted): {f1_score(df['true_label'], df['tuned_pred'], average='weighted'):.4f}\n")
print(classification_report(df["true_label"], df["tuned_pred"]))

if has_baseline:
    print("=" * 70)
    print("📊 KẾT QUẢ BASELINE MODEL (so sánh)")
    print("=" * 70)
    print(f"\nAccuracy : {accuracy_score(df['true_label'], df['base_pred']):.4f}")
    print(f"F1 (weighted): {f1_score(df['true_label'], df['base_pred'], average='weighted'):.4f}\n")
    print(classification_report(df["true_label"], df["base_pred"]))

# ─────────────────────────────────────────
# 7. CONFUSION MATRIX (text)
# ─────────────────────────────────────────
labels_list = sorted(df["true_label"].unique())
cm = confusion_matrix(df["true_label"], df["tuned_pred"], labels=labels_list)
cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels_list],
                          columns=[f"Pred_{l}" for l in labels_list])
print("=" * 70)
print("🔢 CONFUSION MATRIX (Tuned Model)")
print("=" * 70)
print(cm_df.to_string())

# ─────────────────────────────────────────
# 8. PHÂN TÍCH LỖI CHI TIẾT
# ─────────────────────────────────────────
errors = df[~df["tuned_correct"]].copy()
print(f"\n\n{'=' * 70}")
print(f"❌ PHÂN TÍCH LỖI CHI TIẾT  (tổng: {len(errors)}/{len(df)} mẫu)")
print("=" * 70)

# 8a. Loại lỗi phổ biến nhất
print("\n📌 Các cặp (True → Predicted) phổ biến:")
error_pairs = errors.groupby(["true_label", "tuned_pred"]).size().reset_index(name="count")
error_pairs = error_pairs.sort_values("count", ascending=False)
print(error_pairs.to_string(index=False))

# 8b. Độ dài câu của lỗi vs đúng
df["word_count"] = df["text_processed"].str.split().str.len()
print("\n📏 Độ dài câu trung bình (số từ):")
print(f"  Dự đoán đúng  : {df[df['tuned_correct']]['word_count'].mean():.1f} từ")
print(f"  Dự đoán sai   : {df[~df['tuned_correct']]['word_count'].mean():.1f} từ")

# 8c. Ví dụ lỗi từng loại
print("\n📝 Ví dụ lỗi theo từng nhãn thật:")
for true_lbl in labels_list:
    subset = errors[errors["true_label"] == true_lbl].head(3)
    if subset.empty:
        continue
    print(f"\n  [True = {true_lbl}]")
    for _, row in subset.iterrows():
        print(f"    → Pred: {row['tuned_pred']}")
        print(f"       Câu: {row['sentence'][:120]}")

# 8d. Câu ngắn nhất bị sai
short_errors = errors.sort_values("word_count").head(5)
print("\n🔤 5 câu ngắn nhất bị phân loại sai:")
for _, row in short_errors.iterrows():
    print(f"  [{row['word_count']} từ] True={row['true_label']} | Pred={row['tuned_pred']}")
    print(f"  Câu: {row['sentence']}")

# 8e. Câu dài nhất bị sai
long_errors = errors.sort_values("word_count", ascending=False).head(5)
print("\n🔤 5 câu dài nhất bị phân loại sai:")
for _, row in long_errors.iterrows():
    print(f"  [{row['word_count']} từ] True={row['true_label']} | Pred={row['tuned_pred']}")
    print(f"  Câu: {row['sentence'][:150]}")

# ─────────────────────────────────────────
# 9. PHÂN TÍCH ĐỘ TỰ TIN (nếu model hỗ trợ predict_proba)
# ─────────────────────────────────────────
if hasattr(tuned_model, "predict_proba"):
    probs = tuned_model.predict_proba(X_test)
    df["confidence"] = probs.max(axis=1)
    print("\n\n" + "=" * 70)
    print("🎯 PHÂN TÍCH ĐỘ TỰ TIN (Confidence)")
    print("=" * 70)
    print(f"  Confidence trung bình (đúng): {df[df['tuned_correct']]['confidence'].mean():.4f}")
    print(f"  Confidence trung bình (sai) : {df[~df['tuned_correct']]['confidence'].mean():.4f}")

    # Tạo lại errors sau khi đã có cột confidence
    errors = df[~df["tuned_correct"]].copy()

    # High-confidence errors (model sai nhưng rất tự tin)
    high_conf_errors = errors[errors["confidence"] > 0.85].sort_values("confidence", ascending=False)
    print(f"\n⚠️  Lỗi với confidence > 0.85: {len(high_conf_errors)} mẫu")
    for _, row in high_conf_errors.head(5).iterrows():
        print(f"  [{row['confidence']:.2f}] True={row['true_label']} | Pred={row['tuned_pred']}")
        print(f"  Câu: {row['sentence'][:120]}")

    # Low-confidence correct (model đúng nhưng không chắc)
    low_conf_correct = df[df["tuned_correct"] & (df["confidence"] < 0.55)]
    print(f"\n⚡ Đúng nhưng confidence < 0.55: {len(low_conf_correct)} mẫu")

# ─────────────────────────────────────────
# 10. LƯU KẾT QUẢ
# ─────────────────────────────────────────
save_cols = ["sentence", "text_processed", "true_label", "tuned_pred", "tuned_correct", "word_count"]
if "confidence" in df.columns:
    save_cols.append("confidence")
if has_baseline:
    save_cols += ["base_pred", "base_correct"]

df[save_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n\n✅ Đã lưu kết quả phân tích vào: {OUTPUT_CSV}")
print("=" * 70)