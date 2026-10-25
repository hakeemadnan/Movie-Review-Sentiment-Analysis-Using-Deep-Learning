# Movie Review Sentiment Analysis using BiLSTM

A medium-level deep learning project that classifies IMDb movie reviews as positive or negative using a Bidirectional LSTM neural network.

## Features
- Uses a 50,000-review IMDb CSV dataset
- Cleans HTML tags and whitespace
- Uses a stratified train/validation/test split
- Tokenizes text and pads sequences
- Trains a BiLSTM model with dropout and early stopping
- Reports test metrics and saves training plots and a confusion matrix
- Provides an interactive Streamlit web app

## Folder structure
```text
movie_sentiment_lstm_project/
├── IMDB Dataset.csv          # Add your downloaded CSV here
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
├── models/                   # Generated after training
└── outputs/                  # Generated evaluation charts
```

## Setup (Windows PowerShell)
1. Install Python 3.10 or 3.11 if not already installed.
2. Open PowerShell in this folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Copy `IMDB Dataset.csv` into this folder. The file should have columns named `review` and `sentiment`.
6. Train the model:

   ```powershell
   python train_model.py
   ```

7. Launch the web app after training finishes:

   ```powershell
   streamlit run app.py
   ```

## Dataset format
The CSV should contain:
- `review`: movie review text
- `sentiment`: `positive` or `negative`

## Evaluation and report
Use the printed test accuracy and classification report from `train_model.py`. Include `outputs/class_distribution.png`, `outputs/training_curves.png`, and `outputs/confusion_matrix.png` in your report. Do not claim a specific accuracy until your own training run has completed.

## Viva topics
- Why sentiment analysis is an NLP task
- Why an embedding layer is used
- What an LSTM does and why bidirectional LSTM may help
- Why data is split into train, validation, and test sets
- What precision, recall, F1-score, and a confusion matrix mean
- Why early stopping and dropout are used
- Limitations: sarcasm, mixed sentiment, slang, and domain shift
