import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load your data here (assuming base_directory is set)
# texts, labels = load_text_files(your_path) 

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, 'models/model.pkl')
joblib.dump(tfidf, 'models/vectorizer.pkl')
print("Sentiment models saved to models/")
