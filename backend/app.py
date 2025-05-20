import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib

train = pd.read_csv("Dataset/train_clean.csv")
test_clean = pd.read_csv("Dataset/test_clean.csv")
test_labels = pd.read_csv("Dataset/test_labels.csv")

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = tfidf.fit_transform(train['comment_text'])
y_train = train[labels]

model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

joblib.dump(model, "logreg_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

X_test = tfidf.transform(test_clean['comment_text'])
valid = ~(test_labels[labels] == -1).all(axis=1)
y_true = test_labels.loc[valid, labels]
y_pred = model.predict(X_test[valid.values])

print(classification_report(y_true, y_pred, target_names=labels))

def classify_comment(comment: str) -> dict:
    vec = tfidf.transform([comment])
    prediction = model.predict(vec)[0]
    return {label: bool(prediction[i]) for i, label in enumerate(labels)}

comment = input("Enter a comment: ")
print("Prediction:", classify_comment(comment))