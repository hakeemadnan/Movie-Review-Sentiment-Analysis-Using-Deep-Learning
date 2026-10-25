import os
import re
import html
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# -------------------- Configuration --------------------
DATA_PATH = "IMDB Dataset.csv"  # Put your downloaded CSV in this project folder
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
MAX_WORDS = 20000
MAX_LEN = 300
EMBEDDING_DIM = 128
BATCH_SIZE = 128
EPOCHS = 8
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def clean_review(text):
    """Remove HTML tags and normalize whitespace without deleting useful words."""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("1/6 Loading dataset...")
df = pd.read_csv(DATA_PATH)
required_cols = {"review", "sentiment"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns named {required_cols}. Found: {list(df.columns)}")

df = df[["review", "sentiment"]].dropna()
df["review"] = df["review"].apply(clean_review)
df["sentiment"] = df["sentiment"].str.lower().map({"negative": 0, "positive": 1})
df = df.dropna(subset=["sentiment"])
df["sentiment"] = df["sentiment"].astype(int)
df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

print(f"Rows after cleaning: {len(df):,}")
print("Class distribution:")
print(df["sentiment"].map({0: "negative", 1: "positive"}).value_counts())

# Save a class distribution chart for the report
df["sentiment"].map({0: "Negative", 1: "Positive"}).value_counts().reindex(["Negative", "Positive"]).plot(
    kind="bar", title="IMDb Review Sentiment Distribution", rot=0
)
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
plt.close()

X = df["review"].astype(str).to_numpy(dtype=object)
y = df["sentiment"].to_numpy(dtype=np.int32)

print("2/6 Creating train / validation / test splits...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)
print(f"Train: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}")

print("3/6 Converting text into sequences...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def to_padded_sequences(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

X_train_pad = to_padded_sequences(X_train)
X_val_pad = to_padded_sequences(X_val)
X_test_pad = to_padded_sequences(X_test)

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("4/6 Building and training BiLSTM model...")
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.0)),
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(1, activation="sigmoid")
])
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"), monitor="val_loss", save_best_only=True, verbose=1)
]
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("5/6 Evaluating on the held-out test set...")
best_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
test_results = best_model.evaluate(X_test_pad, y_test, verbose=0)
for name, value in zip(best_model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

probabilities = best_model.predict(X_test_pad, batch_size=BATCH_SIZE, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=["Negative", "Positive"], digits=4))
print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.close()

# Save the final best model under a predictable name
best_model.save(os.path.join(MODEL_DIR, "sentiment_model.keras"))
print("\n6/6 Finished!")
print("Saved model: models/sentiment_model.keras")
print("Saved tokenizer: models/tokenizer.pkl")
print("Saved charts in: outputs/")
print("Next: run `streamlit run app.py`")
