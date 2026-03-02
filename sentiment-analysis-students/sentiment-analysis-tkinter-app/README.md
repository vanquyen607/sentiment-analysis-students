# Sentiment Analysis with Tkinter

This project is a sentiment analysis application built using FastAPI for the backend and Tkinter for the graphical user interface (GUI). It allows users to input feedback and receive predictions on sentiment and topic classification.

## Project Structure

```
sentiment-analysis-tkinter
├── src
│   ├── main.py          # Entry point for the application
│   ├── gui.py           # GUI implementation using Tkinter
│   ├── models.py        # Model loading and prediction functions
│   └── utils.py         # Utility functions for text preprocessing
├── tuned_models
│   ├── best_sentiment_model.pkl       # Trained sentiment analysis model
│   ├── optimized_tfidf_sentiment.pkl   # Optimized TF-IDF vectorizer for sentiment
│   ├── best_topic_model.pkl            # Trained topic classification model
│   └── optimized_tfidf_topic.pkl       # Optimized TF-IDF vectorizer for topic
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sentiment-analysis-tkinter
   ```

2. **Install dependencies**:
   Make sure you have Python installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Execute the main script to start the Tkinter GUI:
   ```bash
   python src/main.py
   ```

## Usage

- Once the application is running, you will see a GUI where you can input feedback text.
- After entering the text, click the "Analyze" button to get the sentiment and topic predictions.
- The results will be displayed in the GUI.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.