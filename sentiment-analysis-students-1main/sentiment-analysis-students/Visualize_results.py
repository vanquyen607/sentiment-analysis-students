"""
visualize_results.py
====================
Vẽ toàn bộ biểu đồ phân tích sau khi train mô hình Sentiment Analysis.
Chạy SAU error_analysis.py (đọc file error_analysis_results.csv).

Các biểu đồ được tạo ra:
  1. Confusion Matrix (Tuned vs Baseline nếu có)
  2. Phân phối độ tự tin (confidence) — đúng vs sai
  3. Biểu đồ lỗi theo loại (True→Pred heatmap)
  4. Phân phối độ dài câu — đúng vs sai
  5. Accuracy / F1 per class — bar chart
  6. Confidence threshold curve
"""

import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_recall_fscore_support
)

# ─────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────
RESULTS_CSV       = Path("error_analysis_results.csv")
TUNED_MODEL_PATH  = Path("tuned_models/best_sentiment_model.pkl")
TUNED_TFIDF_PATH  = Path("tuned_models/optimized_tfidf_sentiment.pkl")
BASE_MODEL_PATH   = Path("models/sentiment_model.pkl")
BASE_TFIDF_PATH   = Path("models/tfidf_sentiment.pkl")
TEST_SENT_PATH    = Path("vietnamese_students_feedback/test/sentences.txt")
TEST_LABEL_PATH   = Path("vietnamese_students_feedback/test/sentiments.txt")
OUTPUT_DIR        = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "font.family"    : "DejaVu Sans",
    "axes.titlesize" : 14,
    "axes.labelsize" : 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi"     : 120,
})
PALETTE = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}

# ─────────────────────────────────────────
# TẢI DỮ LIỆU
# ─────────────────────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s\u0080-\u024F\u1E00-\u1EFF.,!?]", " ", text)
    return " ".join(text.split())

# Ưu tiên đọc từ CSV đã tạo bởi error_analysis.py
if RESULTS_CSV.exists():
    df = pd.read_csv(RESULTS_CSV)
    print(f"✓ Đọc từ {RESULTS_CSV}: {len(df)} mẫu")
else:
    # Fallback: load model và predict lại
    print("⚠️  Không tìm thấy error_analysis_results.csv → load model và predict lại...")
    sentences = TEST_SENT_PATH.read_text(encoding="utf-8").strip().splitlines()
    raw_labels = TEST_LABEL_PATH.read_text(encoding="utf-8").strip().splitlines()
    SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive",
                     "0": "negative", "1": "neutral", "2": "positive"}
    def normalize_label(lbl):
        return SENTIMENT_MAP.get(lbl, SENTIMENT_MAP.get(str(lbl), str(lbl)))
    df = pd.DataFrame({"sentence": sentences, "true_label": raw_labels})
    df["true_label"] = df["true_label"].apply(normalize_label)
    df["text_processed"] = df["sentence"].apply(preprocess)
    df["word_count"] = df["text_processed"].str.split().str.len()

    with open(TUNED_MODEL_PATH, "rb") as f: tuned_model = pickle.load(f)
    with open(TUNED_TFIDF_PATH, "rb") as f: tuned_tfidf = pickle.load(f)
    X = tuned_tfidf.transform(df["text_processed"])
    df["tuned_pred"]    = [normalize_label(p) for p in tuned_model.predict(X)]
    df["tuned_correct"] = df["tuned_pred"] == df["true_label"]
    if hasattr(tuned_model, "predict_proba"):
        df["confidence"] = tuned_model.predict_proba(X).max(axis=1)

has_baseline   = "base_pred" in df.columns
has_confidence = "confidence" in df.columns
labels_list    = sorted(df["true_label"].unique())

print(f"   Nhãn: {labels_list}")
print(f"   Có baseline: {has_baseline} | Có confidence: {has_confidence}\n")

# ─────────────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 1 – CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title, ax, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
                       for j in range(len(labels))] for i in range(len(labels))])
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

ncols = 2 if has_baseline else 1
fig1, axes1 = plt.subplots(1, ncols, figsize=(7*ncols, 6))
if not has_baseline:
    axes1 = [axes1]

plot_confusion_matrix(df["true_label"], df["tuned_pred"],
                      "Confusion Matrix – Tuned Model", axes1[0], labels_list)
if has_baseline:
    plot_confusion_matrix(df["true_label"], df["base_pred"],
                          "Confusion Matrix – Baseline Model", axes1[1], labels_list)

plt.suptitle("Confusion Matrix Comparison", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig1.savefig(OUTPUT_DIR / "1_confusion_matrix.png", bbox_inches="tight")
plt.close()
print("✓ Đã lưu: 1_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 2 – PRECISION / RECALL / F1 PER CLASS
# ─────────────────────────────────────────────────────────────────────────────
prec, rec, f1, support = precision_recall_fscore_support(
    df["true_label"], df["tuned_pred"], labels=labels_list)

metrics_df = pd.DataFrame({
    "Class"    : labels_list,
    "Precision": prec,
    "Recall"   : rec,
    "F1_Score" : f1,
    "Support"  : support
})

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart metrics
x = np.arange(len(labels_list))
w = 0.26
for i, (metric, color) in enumerate(zip(["Precision", "Recall", "F1_Score"],
                                          ["#3498db", "#e67e22", "#2ecc71"])):
    axes2[0].bar(x + i*w, metrics_df[metric], w, label=metric,
                 color=color, alpha=0.85, edgecolor="white")

axes2[0].set_xticks(x + w)
axes2[0].set_xticklabels(labels_list)
axes2[0].set_ylim(0, 1.12)
axes2[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
axes2[0].legend()
axes2[0].set_title("Precision / Recall / F1 theo từng nhãn", fontweight="bold")
axes2[0].set_xlabel("Sentiment Class")
for bar_group_start, row in zip(x, metrics_df.itertuples()):
    for j, val in enumerate([row.Precision, row.Recall, row.F1_Score]):
        axes2[0].text(bar_group_start + j*w, val + 0.01, f"{val:.2f}",
                      ha="center", va="bottom", fontsize=9)

# Support (sample count)
bar_colors = [PALETTE.get(l, "#95a5a6") for l in labels_list]
axes2[1].bar(labels_list, metrics_df["Support"], color=bar_colors,
             edgecolor="white", alpha=0.9)
for i, (lbl, sup) in enumerate(zip(labels_list, metrics_df["Support"])):
    axes2[1].text(i, sup + 1, str(sup), ha="center", va="bottom", fontweight="bold")
axes2[1].set_title("Số mẫu theo từng nhãn (Test set)", fontweight="bold")
axes2[1].set_xlabel("Sentiment Class")
axes2[1].set_ylabel("Số mẫu")

plt.suptitle("Performance per Class – Tuned Model", fontsize=15, fontweight="bold")
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / "2_per_class_metrics.png", bbox_inches="tight")
plt.close()
print("✓ Đã lưu: 2_per_class_metrics.png")

# ─────────────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 3 – PHÂN PHỐI LỖI (Error Type Heatmap)
# ─────────────────────────────────────────────────────────────────────────────
errors = df[~df["tuned_correct"]]
error_matrix = pd.crosstab(errors["true_label"], errors["tuned_pred"],
                            rownames=["True"], colnames=["Predicted"])
error_matrix = error_matrix.reindex(index=labels_list, columns=labels_list, fill_value=0)
np.fill_diagonal(error_matrix.values, 0)  # không cần ô đúng

fig3, ax3 = plt.subplots(figsize=(7, 5))
sns.heatmap(error_matrix, annot=True, fmt="d", cmap="Reds",
            linewidths=0.5, linecolor="white", ax=ax3,
            cbar_kws={"label": "Số lỗi"})
ax3.set_title("Error Distribution: True → Predicted\n(chỉ các trường hợp sai)",
               fontweight="bold")
plt.tight_layout()
fig3.savefig(OUTPUT_DIR / "3_error_heatmap.png", bbox_inches="tight")
plt.close()
print("✓ Đã lưu: 3_error_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 4 – ĐỘ DÀI CÂU: Đúng vs Sai
# ─────────────────────────────────────────────────────────────────────────────
if "word_count" not in df.columns:
    df["word_count"] = df["text_processed"].str.split().str.len()

fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))

# Histogram
for is_correct, label, color in [(True, "Đúng", "#2ecc71"), (False, "Sai", "#e74c3c")]:
    subset = df[df["tuned_correct"] == is_correct]["word_count"]
    axes4[0].hist(subset, bins=25, alpha=0.6, color=color, label=label, edgecolor="white")
axes4[0].set_xlabel("Số từ trong câu")
axes4[0].set_ylabel("Số mẫu")
axes4[0].set_title("Phân phối độ dài câu (Đúng vs Sai)", fontweight="bold")
axes4[0].legend()
axes4[0].axvline(df[df["tuned_correct"]]["word_count"].mean(),
                 color="#27ae60", linestyle="--", linewidth=1.5, label="Mean đúng")
axes4[0].axvline(df[~df["tuned_correct"]]["word_count"].mean(),
                 color="#c0392b", linestyle="--", linewidth=1.5, label="Mean sai")

# Boxplot theo nhãn
box_data = [df[(df["true_label"]==l) & (df["tuned_correct"])]["word_count"].tolist() for l in labels_list]
box_err  = [df[(df["true_label"]==l) & (~df["tuned_correct"])]["word_count"].tolist() for l in labels_list]
positions = np.arange(len(labels_list))
bp1 = axes4[1].boxplot(box_data,  positions=positions-0.2, widths=0.35,
                        patch_artist=True, boxprops=dict(facecolor="#2ecc71", alpha=0.7),
                        medianprops=dict(color="black", linewidth=2))
bp2 = axes4[1].boxplot(box_err,   positions=positions+0.2, widths=0.35,
                        patch_artist=True, boxprops=dict(facecolor="#e74c3c", alpha=0.7),
                        medianprops=dict(color="black", linewidth=2))
axes4[1].set_xticks(positions)
axes4[1].set_xticklabels(labels_list)
axes4[1].set_xlabel("Sentiment Class")
axes4[1].set_ylabel("Số từ")
axes4[1].set_title("Độ dài câu theo nhãn (xanh=đúng, đỏ=sai)", fontweight="bold")
axes4[1].legend([bp1["boxes"][0], bp2["boxes"][0]], ["Đúng", "Sai"])

plt.suptitle("Phân tích độ dài câu và lỗi", fontsize=15, fontweight="bold")
plt.tight_layout()
fig4.savefig(OUTPUT_DIR / "4_length_analysis.png", bbox_inches="tight")
plt.close()
print("✓ Đã lưu: 4_length_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 5 – CONFIDENCE DISTRIBUTION (nếu có)
# ─────────────────────────────────────────────────────────────────────────────
if has_confidence:
    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))

    # 5a. KDE: đúng vs sai
    from scipy.stats import gaussian_kde
    x_range = np.linspace(0, 1, 300)
    for is_correct, label, color in [(True, "Đúng", "#2ecc71"), (False, "Sai", "#e74c3c")]:
        subset = df[df["tuned_correct"] == is_correct]["confidence"].dropna().values
        kde = gaussian_kde(subset)
        y_kde = kde(x_range)
        axes5[0].plot(x_range, y_kde, color=color, label=label, linewidth=2)
        axes5[0].fill_between(x_range, 0, y_kde, color=color, alpha=0.15)
    axes5[0].set_xlim(0, 1)
    axes5[0].set_title("Phân phối Confidence\n(Đúng vs Sai)", fontweight="bold")
    axes5[0].set_xlabel("Confidence")
    axes5[0].legend()

    # 5b. Accuracy theo threshold
    thresholds = np.linspace(0.4, 0.99, 60)
    accs, coverages = [], []
    for t in thresholds:
        subset = df[df["confidence"] >= t]
        if len(subset) == 0:
            accs.append(np.nan); coverages.append(0)
        else:
            accs.append(accuracy_score(subset["true_label"], subset["tuned_pred"]))
            coverages.append(len(subset) / len(df))

    ax5b = axes5[1]
    color1, color2 = "#3498db", "#e67e22"
    l1, = ax5b.plot(thresholds, accs, color=color1, linewidth=2, label="Accuracy")
    ax5b.set_xlabel("Confidence Threshold")
    ax5b.set_ylabel("Accuracy", color=color1)
    ax5b.tick_params(axis="y", labelcolor=color1)
    ax5b2 = ax5b.twinx()
    l2, = ax5b2.plot(thresholds, [c*100 for c in coverages], color=color2,
                      linewidth=2, linestyle="--", label="Coverage %")
    ax5b2.set_ylabel("Coverage (%)", color=color2)
    ax5b2.tick_params(axis="y", labelcolor=color2)
    ax5b.set_title("Accuracy vs Coverage\ntheo Confidence Threshold", fontweight="bold")
    ax5b.legend(handles=[l1, l2], loc="lower left")

    # 5c. Confidence per class
    df.boxplot(column="confidence", by="true_label", ax=axes5[2],
               patch_artist=True)
    axes5[2].set_title("Confidence theo từng nhãn", fontweight="bold")
    axes5[2].set_xlabel("Sentiment Class")
    axes5[2].set_ylabel("Confidence")
    plt.sca(axes5[2])
    plt.xticks(rotation=0)

    plt.suptitle("Phân tích Confidence Score", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig5.savefig(OUTPUT_DIR / "5_confidence_analysis.png", bbox_inches="tight")
    plt.close()
    print("✓ Đã lưu: 5_confidence_analysis.png")
else:
    print("⚠️  Bỏ qua biểu đồ Confidence (model không có predict_proba)")

# ─────────────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 6 – TỔNG HỢP SO SÁNH (nếu có baseline)
# ─────────────────────────────────────────────────────────────────────────────
if has_baseline:
    models = {
        "Baseline": (df["true_label"], df["base_pred"]),
        "Tuned"   : (df["true_label"], df["tuned_pred"]),
    }
    summary = []
    for mname, (y_true, y_pred) in models.items():
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        summary.append({"Model": mname,
                         "Accuracy" : accuracy_score(y_true, y_pred),
                         "Precision": p, "Recall": r, "F1": f})
    summary_df = pd.DataFrame(summary)

    fig6, ax6 = plt.subplots(figsize=(8, 5))
    x   = np.arange(len(summary_df))
    w   = 0.2
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1"]
    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    for i, (metric, color) in enumerate(zip(metrics_cols, colors)):
        bars = ax6.bar(x + i*w, summary_df[metric], w, label=metric,
                       color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax6.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=8)

    ax6.set_xticks(x + 1.5*w)
    ax6.set_xticklabels(summary_df["Model"], fontsize=12)
    ax6.set_ylim(0, 1.12)
    ax6.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax6.set_title("So sánh Baseline vs Tuned Model (tổng hợp)", fontweight="bold")
    ax6.legend()
    plt.tight_layout()
    fig6.savefig(OUTPUT_DIR / "6_model_comparison.png", bbox_inches="tight")
    plt.close()
    print("✓ Đã lưu: 6_model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# TỔNG KẾT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"🎨 Tất cả biểu đồ đã lưu vào thư mục: {OUTPUT_DIR}/")
saved = sorted(OUTPUT_DIR.glob("*.png"))
for p in saved:
    print(f"   • {p.name}")
print("=" * 60)