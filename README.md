# AffectiveInsight: Multi-Engine Sentiment & Emotion Analytics

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/framework-Flask-lightgrey)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

AffectiveInsight is a sophisticated Natural Language Processing (NLP) dashboard designed for dual-layer text analysis. By combining probabilistic statistical modeling with deep emotion classification, the platform transforms raw text into actionable emotional intelligence through real-time data visualization.

---

## 📍 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Technical Insights](#-technical-insights)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Project Overview
AffectiveInsight bridges the gap between simple "Positive/Negative" sentiment and complex human emotion. It utilizes a **Logistic Regression** engine for sentiment scoring and a **Multi-class Classifier** to map text across six core emotional states. The results are rendered via an interactive Flask-based dashboard featuring dynamic Matplotlib visualizations.

---

## 🚀 Key Features
*   **Probabilistic Sentiment Scoring:** Employs Logistic Regression to provide confidence scores, enabling a nuanced "Neutral" classification instead of binary outputs.
*   **Granular Emotion Mapping:** Detects six distinct emotional states: `Anger`, `Fear`, `Joy`, `Love`, `Sadness`, and `Surprise`.
*   **Dynamic Visual Analytics:** Generates real-time bar charts of emotion probabilities, delivered instantly to the UI via Base64 encoding.
*   **Hybrid NLP Pipeline:** Integrates **Scikit-Learn** for statistical modeling and **NLTK** for advanced lemmatization and VADER-based validation.
*   **AJAX-Enabled Dashboard:** A responsive frontend that communicates with the Flask backend via the Fetch API for a seamless, no-refresh user experience.

---

## 🛠️ Tech Stack
| Category | Technology |
| :--- | :--- |
| **Web Framework** | Flask |
| **Machine Learning** | Scikit-Learn (Logistic Regression, TF-IDF) |
| **NLP Utilities** | NLTK (WordNetLemmatizer, VADER, Tokenization) |
| **Visualization** | Matplotlib, IO, Base64 |
| **Serialization** | Joblib, Pickle |

---

## 📁 Project Structure
```plaintext
├── app.py                      # Flask Server with Visualization Engine
├── Sentiment_Analysis_LR.py    # Logistic Regression Training Script
├── Emotion_Detection_Final.py  # Multi-class Emotion Model Training
├── NLTK_Exploration.py         # Lexicon-based Emotion & Sentiment Analysis
├── models/
│   ├── model.pkl               # Trained Sentiment Classifier
│   ├── vectorizer.pkl          # Sentiment TF-IDF Vectorizer
│   ├── emotion_model.pkl       # Trained Emotion Classifier
│   └── emotion_vectorizer.pkl  # Emotion TF-IDF Vectorizer
├── templates/
│   └── index.html              # Interactive AJAX-enabled Dashboard
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites
*   Python 3.8 or higher
*   pip (Python package manager)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/AffectiveInsight.git
    cd AffectiveInsight
    ```

2.  **Install dependencies:**
    ```bash
    pip install flask joblib scikit-learn nltk matplotlib pandas
    ```

3.  **Download NLTK resources:**
    ```python
    python -c "import nltk; nltk.download(['stopwords', 'wordnet', 'vader_lexicon', 'punkt'])"
    ```

### Running the Application
Launch the Flask dashboard with the following command:
```bash
python app.py
```
Once the server starts, navigate to `http://127.0.0.1:5000` in your web browser.

---

## 💡 Technical Insights

### Thread-Safe Visualization
To ensure stability in a web environment, AffectiveInsight utilizes Matplotlib’s **Object-Oriented API**. This avoids the common "Image Overlap" issue in Flask caused by the stateful nature of `pyplot`.
```python
# Implementation approach
from matplotlib.figure import Figure
fig = Figure()
ax = fig.subplots()
ax.bar(emotions, probabilities)
```

### Robust Confidence Thresholding
The engine implements a thresholding logic ($0.4$ to $0.6$) to determine sentiment. Future iterations are set to include **Softmax-based confidence checks**; if the highest probability falls below a $0.35$ threshold, the input is flagged as "Ambiguous" to maintain data integrity.

### Real-time Integration
The frontend utilizes the **Fetch API** to send JSON requests to the `/predict` endpoint. The backend returns the sentiment analysis and a Base64 encoded string of the Matplotlib plot, allowing for instantaneous UI updates without page reloads.

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the model accuracy or UI/UX:
1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
**Developed by [Your Name/Organization]** - *Turning text into insights.*