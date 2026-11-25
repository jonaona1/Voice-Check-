import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

LABELS = ["Register", "Repeat", "Cancel", "Help"]

# 1. Load data (drop duplicates so test phrases are never also in training)
df = pd.read_csv("phrases.csv", encoding="utf-8")
df["text"] = df["text"].str.lower().str.strip()
df = df.drop_duplicates(subset="text")

# 2. Split: 80% train, 20% hidden test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
print(f"Train: {len(X_train)} phrases, Test: {len(X_test)} phrases\n")

# 3. Three models, same TF-IDF (character n-grams cope with misheard words)
def tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))

models = {
    "Naive Bayes": make_pipeline(tfidf(), MultinomialNB()),
    "Logistic Regression": make_pipeline(tfidf(), LogisticRegression(max_iter=1000)),
    "SVM": make_pipeline(tfidf(), LinearSVC()),
}

# 4. Train and measure
results, preds = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, p, average="macro", zero_division=0)
    results.append((name, accuracy_score(y_test, p), prec, rec, f1))

print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
for name, a, pr, rc, f in results:
    print(f"{name:<22}{a:>10.3f}{pr:>11.3f}{rc:>9.3f}{f:>8.3f}")

# 5. Best model, confusion matrix, mistakes
best = max(results, key=lambda r: r[4])[0]
print(f"\nBest model by F1: {best}\n")
cm = confusion_matrix(y_test, preds[best], labels=LABELS)
print("Confusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS), "\n")
wrong = [(t, a, p) for t, a, p in zip(X_test, y_test, preds[best]) if a != p]
print("Phrases the best model got wrong:" if wrong else "No mistakes on the test phrases.")
for t, a, p in wrong:
    print(f'  "{t}"  true={a}  predicted={p}')

# 6. Live typed demo
model = models[best]
print("\nType a phrase to classify it (empty line to quit):")
while True:
    text = input("> ").strip().lower()
    if not text:
        break
    print("   ->", model.predict([text])[0])