# 🔎 **ReviewClassifier AI — Fake Review Detection System**

An NLP + Machine Learning–powered web application that detects whether a review is **Fake** or **Genuine**.
Built using **Python, Scikit-Learn, TF-IDF, Voting Classifier, and Streamlit**.

---

## 💻 **Live Demo**

*Coming Soon* 

---

## 📂 **Project Repository**

Source code:
🔗 [https://github.com/Aswin-MS/Fake-Review-Prediction](Code)

---

## 📖 **Table of Contents**

* Overview
* Features
* System Architecture
* Machine Learning Model
* Dataset
* Installation
* Usage
* App Interface
* Project Structure
* Future Enhancements
* Contributing
* License
* Contact

---

## 📝 **Overview**

Fake reviews are increasingly common on e-commerce platforms, influencing customer decisions and damaging marketplace credibility.
**ReviewClassifier AI** provides an automated way to classify reviews using advanced NLP and ML techniques.
The system supports **single review analysis**, **batch CSV prediction**, **threshold control**, and **probability-based outputs**, offering transparency and flexibility to users.

---

## ⭐ **Features**

* Classifies reviews as **Fake** or **Genuine**
* Shows **prediction probability**
* Adjustable **threshold slider** to control strictness
* **Batch prediction** using CSV upload
* Automatic **text-column detection**
* Clean and interactive **Streamlit UI**
* Downloadable results in CSV format
* Ensemble model (Voting Classifier) for stable performance

---

## 🧠 **Machine Learning Model**

The model uses a **Voting Classifier (soft voting)** combining:

| Model                       | Strength                          |
| --------------------------- | --------------------------------- |
| **Logistic Regression**     | Good for linear text patterns     |
| **Multinomial Naive Bayes** | Strong for frequency-based NLP    |
| **Random Forest**           | Captures non-linear relationships |

The ensemble outputs a **probability score** for how likely a review is fake.
This value is compared against a user-defined threshold to generate the final label.

**Text Vectorization:**

* TF-IDF (Term Frequency–Inverse Document Frequency)

**Preprocessing:**

* Lowercasing
* Removing punctuation
* Removing stopwords
* Lemmatization

---

## 🏗️ **System Architecture**

```
User Input (Text / CSV)
        ↓
Text Preprocessing (cleaning, stopwords removal, lemmatization)
        ↓
Feature Extraction (TF-IDF Vectorization)
        ↓
Voting Classifier (LR + Naive Bayes + Random Forest)
        ↓
Post-processing (Thresholding)
        ↓
Output (Fake/Genuine + Probability)
```

Frontend: **Streamlit**
Model Storage: `app/models/model.pkl` & `vectorizer.pkl`

---

## 📊 **Dataset**

* Dataset contains labeled **fake** and **genuine** product reviews.
* Preprocessing performed using a custom cleaning pipeline.
  (Add dataset link here if publicly available)

---

## 🛠️ **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/Aswin-MS/Fake-Review-Prediction.git
cd Fake-Review-Prediction
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the App**

```bash
streamlit run app/app.py
```

---

## 📌 **Usage**

### **🔹 Single Review Prediction**

Enter or paste a review → Click **Predict** → Get:

* Fake / Genuine
* Probability score

### **🔹 Batch Prediction**

Upload a CSV → Auto-detects text column → Runs classification →
Download results as CSV.

### **🔹 Threshold Slider**

Set how strict the system should be:

* **Higher threshold** → fewer reviews marked fake
* **Lower threshold** → more sensitive detection

---

## 🖼️ **App Interface (Screenshots)**

(Add screenshots here)

Example sections:

* Home Screen / Title
* Single Review Input
* Threshold Slider
* Results
* Batch CSV Upload
* Predictions Table

---

## 📁 **Project Structure**

```
Fake-Review-Prediction/
│
├── app/
│   ├── app.py              
│   └── models/
│         ├── model.pkl        
│         └── vectorizer.pkl   
├── notebooks/               
|   ├──fake_reviews.csv
|   └──fake-review-detection.ipynb
├── requirements.txt
|   └──fake_reviews.csv
├── README.md
└── data/                   
```

---

## 🔮 **Future Enhancements**

* Integrate **URL-based features** to detect promotional or suspicious links inside reviews
* Upgrade model using **BERT, RoBERTa, or DistilBERT** for deeper semantic understanding
* Add **multilingual support** for Indian regional languages
* Add **sentiment analysis** along with fake review detection
* Deploy as **REST API** using FastAPI
* Add dashboard analytics (fake review trends, domain stats, etc.)

---

## 🤝 **Contributing**

Pull requests are welcome.
Feel free to open issues for suggestions or improvements.

---

## 📄 **License**

MIT License (or whichever you choose)

---

## 📬 **Contact**

**Aswin M S**

* GitHub: [https://github.com/Aswin-MS](https://github.com/Aswin-MS)
* LinkedIn: [https://www.linkedin.com/in/aswinms175](https://www.linkedin.com/in/aswinms175)
* Email: [msaswin175@gmail.com](mailto:msaswin175@gmail.com)

---

# 🎉 Your README.md is ready!

If you want:

✅ A shorter README
✅ A more visual README with shields/badges
✅ Markdown tables for features
✅ A version with emojis removed

Just tell me — I can generate any version you prefer.
