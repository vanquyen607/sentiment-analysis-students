import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk, filedialog
import pickle
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ttkthemes import ThemedStyle
import threading
import csv

# Load models (tương tự main.py)
models_dir = Path("tuned_models")
with open(models_dir / 'best_sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)
with open(models_dir / 'optimized_tfidf_sentiment.pkl', 'rb') as f:
    sentiment_vectorizer = pickle.load(f)
with open(models_dir / 'best_topic_model.pkl', 'rb') as f:
    topic_model = pickle.load(f)
with open(models_dir / 'optimized_tfidf_topic.pkl', 'rb') as f:
    topic_vectorizer = pickle.load(f)

# Preprocess function (từ main.py)
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s\u0080-\u024F\u1E00-\u1EFF.,!?]', ' ', text)
    text = ' '.join(text.split())
    return text

# Prediction functions (từ main.py)
def predict_sentiment(text: str):
    processed_text = preprocess_text(text)
    text_vec = sentiment_vectorizer.transform([processed_text])
    prediction = sentiment_model.predict(text_vec)[0]
    confidence = None
    if hasattr(sentiment_model, 'predict_proba'):
        proba = sentiment_model.predict_proba(text_vec)[0]
        confidence = float(proba[prediction])
    sentiment_labels = ['negative', 'neutral', 'positive']
    return {'sentiment': sentiment_labels[prediction], 'confidence': confidence}

def predict_topic(text: str):
    processed_text = preprocess_text(text)
    text_vec = topic_vectorizer.transform([processed_text])
    prediction = topic_model.predict(text_vec)[0]
    confidence = None
    if hasattr(topic_model, 'predict_proba'):
        proba = topic_model.predict_proba(text_vec)[0]
        confidence = float(proba[prediction])
    topic_labels = ['lecturer', 'training_program', 'facility', 'others']
    return {'topic': topic_labels[prediction], 'confidence': confidence}

# GUI Class with additional improvements
class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vietnamese Students Feedback Classifier v4.0")
        self.root.geometry("1000x700")

        # Apply theme
        self.style = ThemedStyle(root)
        self.style.set_theme("arc")  # Default light theme
        self.dark_mode = False

        # Menu bar
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear", command=self.clear)
        file_menu.add_command(label="Export Batch to CSV", command=self.export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Toggle Dark Mode", command=self.toggle_dark_mode)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.about)

        # Notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Single prediction tab
        self.single_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.single_frame, text="Single Prediction")
        self.setup_single_tab()

        # Batch prediction tab
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="Batch Prediction")
        self.setup_batch_tab()

        # Visualization tab
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        self.setup_viz_tab()

    def setup_single_tab(self):
        # Input with validation
        ttk.Label(self.single_frame, text="Nhập feedback (tối thiểu 10 ký tự):").pack(pady=5)
        self.input_text = scrolledtext.ScrolledText(self.single_frame, height=5, wrap=tk.WORD)
        self.input_text.pack(pady=5, padx=10, fill=tk.X)

        # Predict button
        self.predict_btn = ttk.Button(self.single_frame, text="Dự đoán", command=self.predict_single)
        self.predict_btn.pack(pady=10)
        self.create_tooltip(self.predict_btn, "Nhấn để dự đoán sentiment và topic")

        # Output
        ttk.Label(self.single_frame, text="Kết quả:").pack(pady=5)
        self.output_text = scrolledtext.ScrolledText(self.single_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

    def setup_batch_tab(self):
        # Input
        ttk.Label(self.batch_frame, text="Nhập nhiều feedback (mỗi dòng một câu):").pack(pady=5)
        self.batch_input = scrolledtext.ScrolledText(self.batch_frame, height=10, wrap=tk.WORD)
        self.batch_input.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # Buttons and progress
        button_frame = ttk.Frame(self.batch_frame)
        button_frame.pack(pady=10)
        self.batch_btn = ttk.Button(button_frame, text="Dự đoán Batch", command=self.start_batch_thread)
        self.batch_btn.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.batch_btn, "Dự đoán cho nhiều câu")
        ttk.Button(button_frame, text="Load từ File", command=self.load_file).pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(self.batch_frame, orient="horizontal", mode="determinate")
        self.progress.pack(pady=5, padx=10, fill=tk.X)

        # Search
        search_frame = ttk.Frame(self.batch_frame)
        search_frame.pack(pady=5)
        ttk.Label(search_frame, text="Tìm kiếm:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Filter", command=self.filter_tree).pack(side=tk.LEFT)

        # Output Treeview
        self.tree = ttk.Treeview(self.batch_frame, columns=("Original", "Processed", "Sentiment", "Confidence_S", "Topic", "Confidence_T"), show="headings")
        self.tree.heading("Original", text="Original")
        self.tree.heading("Processed", text="Processed")
        self.tree.heading("Sentiment", text="Sentiment")
        self.tree.heading("Confidence_S", text="Conf_S")
        self.tree.heading("Topic", text="Topic")
        self.tree.heading("Confidence_T", text="Conf_T")
        self.tree.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # Scrollbar for tree
        scrollbar = ttk.Scrollbar(self.batch_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_viz_tab(self):
        ttk.Label(self.viz_frame, text="Biểu đồ phân phối Sentiment từ Batch:").pack(pady=5)
        self.viz_btn = ttk.Button(self.viz_frame, text="Vẽ Biểu Đồ", command=self.draw_chart)
        self.viz_btn.pack(pady=10)
        self.create_tooltip(self.viz_btn, "Vẽ biểu đồ từ kết quả batch")
        self.canvas_frame = ttk.Frame(self.viz_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def predict_single(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if len(text) < 10:
            messagebox.showerror("Lỗi", "Feedback phải ít nhất 10 ký tự!")
            return

        try:
            sentiment = predict_sentiment(text)
            topic = predict_topic(text)
            processed = preprocess_text(text)

            result = f"Original: {text}\nProcessed: {processed}\n\nSentiment: {sentiment['sentiment']} (Confidence: {sentiment['confidence'] or 'N/A'})\nTopic: {topic['topic']} (Confidence: {topic['confidence'] or 'N/A'})"

            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, result)
            self.output_text.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi dự đoán: {str(e)}")

    def start_batch_thread(self):
        threading.Thread(target=self.predict_batch).start()

    def predict_batch(self):
        texts = self.batch_input.get("1.0", tk.END).strip().split('\n')
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            messagebox.showerror("Lỗi", "Vui lòng nhập ít nhất một feedback!")
            return

        try:
            self.progress["maximum"] = len(texts)
            self.progress["value"] = 0
            for item in self.tree.get_children():
                self.tree.delete(item)

            self.sentiments = []  # For viz
            self.batch_results = []  # For export
            for i, text in enumerate(texts):
                sentiment = predict_sentiment(text)
                topic = predict_topic(text)
                processed = preprocess_text(text)
                conf_s = f"{sentiment['confidence']:.2f}" if sentiment['confidence'] else "N/A"
                conf_t = f"{topic['confidence']:.2f}" if topic['confidence'] else "N/A"

                # Color based on sentiment
                color = "green" if sentiment['sentiment'] == "positive" else "red" if sentiment['sentiment'] == "negative" else "blue"
                self.tree.insert("", tk.END, values=(text, processed, sentiment['sentiment'], conf_s, topic['topic'], conf_t), tags=(color,))
                self.tree.tag_configure(color, foreground=color)
                self.sentiments.append(sentiment['sentiment'])
                self.batch_results.append((text, processed, sentiment['sentiment'], conf_s, topic['topic'], conf_t))
                self.progress["value"] = i + 1
                self.root.update_idletasks()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi dự đoán batch: {str(e)}")
        finally:
            self.progress["value"] = 0

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.batch_input.delete("1.0", tk.END)
            self.batch_input.insert(tk.END, content)

    def draw_chart(self):
        if not hasattr(self, 'sentiments') or not self.sentiments:
            messagebox.showerror("Lỗi", "Chạy batch prediction trước!")
            return

        # Clear previous chart
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots()
        labels = ['negative', 'neutral', 'positive']
        counts = [self.sentiments.count(l) for l in labels]
        ax.bar(labels, counts, color=['red', 'blue', 'green'])
        ax.set_title("Phân Phối Sentiment")
        ax.set_ylabel("Số lượng")

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_csv(self):
        if not hasattr(self, 'batch_results') or not self.batch_results:
            messagebox.showerror("Lỗi", "Chạy batch prediction trước!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Original", "Processed", "Sentiment", "Confidence_S", "Topic", "Confidence_T"])
                writer.writerows(self.batch_results)
            messagebox.showinfo("Thành công", "Xuất CSV thành công!")

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        theme = "equilux" if self.dark_mode else "arc"
        self.style.set_theme(theme)

    def filter_tree(self):
        query = self.search_var.get().lower()
        for item in self.tree.get_children():
            values = self.tree.item(item, 'values')
            if query in ' '.join(values).lower():
                self.tree.item(item, open=True)
            else:
                self.tree.detach(item)

    def clear(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.batch_input.delete("1.0", tk.END)
        for item in self.tree.get_children():
            self.tree.delete(item)
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        self.progress["value"] = 0
        self.search_var.set("")

    def about(self):
        messagebox.showinfo("About", "Vietnamese Students Feedback Classifier v4.0\nBuilt with Tkinter, ML models, and Matplotlib.")

    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="yellow", relief="solid", borderwidth=1)
            label.pack()
            def hide_tooltip():
                tooltip.destroy()
            widget.tooltip = tooltip
            widget.bind("<Leave>", lambda e: hide_tooltip())
        widget.bind("<Enter>", show_tooltip)

# Run app
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()