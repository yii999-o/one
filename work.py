import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
y_pred_nb = nb_classifier.predict(X_val_tfidf)
print("Naive Bayes Accuracy:", accuracy_score(y_val, y_pred_nb))

svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)
y_pred_svm = svm_classifier.predict(X_val_tfidf)
print("SVM Accuracy:", accuracy_score(y_val, y_pred_svm))

lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train_tfidf, y_train)
y_pred_lr = lr_classifier.predict(X_val_tfidf)
print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_classifier.fit(X_train_tfidf, y_train)
y_pred_mlp = mlp_classifier.predict(X_val_tfidf)
print("MLP Accuracy:", accuracy_score(y_val, y_pred_mlp))
