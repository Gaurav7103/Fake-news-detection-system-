import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from pathlib import Path
import numpy as np
import re
from collections import Counter

DATA_DIR = Path("data")

def load_and_balance_data():
    """Load and properly balance the dataset"""
    try:
        # Load datasets
        real_df = pd.read_csv(DATA_DIR / "Real.csv")
        fake_df = pd.read_csv(DATA_DIR / "Fake.csv")

        print(f"Original data - Real: {len(real_df)}, Fake: {len(fake_df)}")

        # Find text columns
        real_text_col = find_text_column(real_df)
        fake_text_col = find_text_column(fake_df)

        # Extract texts
        real_texts = real_df[real_text_col].dropna().astype(str).tolist()
        fake_texts = fake_df[fake_text_col].dropna().astype(str).tolist()

        # Clean texts
        real_texts = [clean_text(text) for text in real_texts if len(clean_text(text)) > 20]
        fake_texts = [clean_text(text) for text in fake_texts if len(clean_text(text)) > 20]

        print(f"After cleaning - Real: {len(real_texts)}, Fake: {len(fake_texts)}")

        # Balance dataset (take minimum of both)
        min_samples = min(len(real_texts), len(fake_texts))
        print(f"Balancing to {min_samples} samples per class")

        # Create balanced dataset
        texts = real_texts[:min_samples] + fake_texts[:min_samples]
        labels = ['real'] * min_samples + ['fake'] * min_samples

        # Create DataFrame and shuffle
        df = pd.DataFrame({'text': texts, 'label': labels})
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Final balanced dataset: {len(df)} samples")
        print(f"Class distribution: {df['label'].value_counts().to_dict()}")

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic balanced data if files not available"""
    print("Creating synthetic balanced dataset...")

    # Real news samples (factual, neutral language)
    real_samples = [
        "The government announced new economic policies to boost growth in the manufacturing sector and create jobs.",
        "Scientists discovered new marine species during a research expedition in the Pacific Ocean depths.",
        "Education department launched digital learning platform to improve access for students across regions.",
        "Healthcare authorities reported significant improvements in patient care and hospital infrastructure.",
        "International climate conference reached agreement on reducing carbon emissions by thirty percent.",
        "Economic indicators show steady growth with inflation remaining under control this quarter.",
        "New public transportation system will connect major cities reducing travel time significantly.",
        "Research study confirms benefits of regular exercise for cardiovascular health and longevity.",
        "Technology companies announced partnership to develop sustainable energy solutions for data centers.",
        "Agricultural department introduced new techniques to increase crop yield and reduce water usage."
    ]

    # Fake news samples (sensational, emotional language)
    fake_samples = [
        "SHOCKING: Government hiding ALIEN technology that will CHANGE everything you know!",
        "URGENT: Miracle cure discovered for cancer but BIG PHARMA suppressing it! DELETE NOW!",
        "BREAKING: Your phone is SPYING on you - immediate action required! WARNING!",
        "WORLD ENDING TOMORROW: Scientists confirm asteroid impact - government silent!",
        "EXPOSED: Celebrity secret that will SHOCK the world - they don't want you to know!",
        "EMERGENCY: Bank accounts will be FROZEN - withdraw money immediately! ALERT!",
        "UNBELIEVABLE: Simple trick to lose weight without exercise - doctors hate this!",
        "DANGEROUS: Food item in your kitchen causing cancer - remove immediately! WARNING!",
        "SECRET: Ancient remedy cures all diseases - medical industry terrified! SHARE!",
        "CRITICAL: Major event being hidden from public - spread this message quickly!"
    ]

    # Create balanced dataset (500 each)
    texts = real_samples * 50 + fake_samples * 50
    labels = ['real'] * 500 + ['fake'] * 500

    df = pd.DataFrame({'text': texts, 'label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Synthetic dataset created: {len(df)} samples")
    return df

def find_text_column(df):
    """Find the text column in dataframe"""
    for col in df.columns:
        if df[col].dtype == 'object' and len(df[col]) > 0:
            return col
    return df.columns[0]

def clean_text(text):
    """Enhanced text cleaning"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?,]', '', text)

    return text.strip()

def train_balanced_model():
    """Train model with balanced data and bias correction"""
    print("🚀 Training Balanced Fake News Detection Model")
    print("=" * 50)

    # Load balanced data
    df = load_and_balance_data()

    # Prepare features and labels
    X = df['text']
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Train class distribution: {Counter(y_train)}")
    print(f"Test class distribution: {Counter(y_test)}")

    # Enhanced TF-IDF with n-grams to catch fake news patterns
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),  # Include trigrams to catch sensational phrases
        stop_words='english',
        min_df=2,
        max_df=0.85,
        strip_accents='unicode'
    )

    # Logistic Regression with class weights and bias towards fake news detection
    model = LogisticRegression(
        class_weight={'real': 1, 'fake': 1.2},  # Slightly higher weight for fake class
        random_state=42,
        max_iter=1000,
        C=0.8,  # More regularization to prevent overfitting
        solver='liblinear'
    )

    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', model)
    ])

    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Test with known examples
    print("\n🧪 TESTING WITH KNOWN EXAMPLES:")
    print("=" * 40)

    test_cases = [
        # Fake news examples
        ("BREAKING: ALIENS LANDED IN DELHI GOVERNMENT HIDING THE TRUTH URGENT", "fake"),
        ("MIRACLE CURE FOR CANCER DISCOVERED DOCTORS IN SHOCK DELETE NOW", "fake"),
        ("EMERGENCY YOUR PHONE IS SPYING ON YOU TAKE IMMEDIATE ACTION", "fake"),
        ("WORLD ENDING TOMORROW SCIENTISTS CONFIRM GOVERNMENT SILENT", "fake"),

        # Real news examples
        ("The government announced new policies for economic development", "real"),
        ("Scientists discovered new species in ocean research expedition", "real"),
        ("Education department implemented new learning programs for students", "real"),
        ("Healthcare authorities reported improvement in medical facilities", "real"),
    ]

    correct_predictions = 0
    for text, expected in test_cases:
        prediction = pipeline.predict([text])[0]
        probability = pipeline.predict_proba([text])[0]
        is_correct = prediction == expected

        if is_correct:
            correct_predictions += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} Text: {text}")
        print(f"   Expected: {expected}, Predicted: {prediction}")
        print(f"   Probabilities: Real {probability[0]:.3f}, Fake {probability[1]:.3f}")
        print()

    print(f"Test Accuracy: {correct_predictions/len(test_cases)*100:.1f}%")

    # Save model
    joblib.dump(pipeline, "balanced_model.pkl")
    print("💾 Model saved as: balanced_model.pkl")

    return pipeline

# Train the model
if __name__ == "__main__":
    model = train_balanced_model()
