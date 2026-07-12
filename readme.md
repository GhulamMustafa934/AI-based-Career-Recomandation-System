<div align="center">

# 🧭 Career Path Recommender

### An ML-powered web app that reads a resume and predicts the best career paths

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#license)

*Upload a resume, and let 3 trained machine learning models tell you which career paths fit best.*

</div>

---

## ✨ What It Does

This app takes a resume as input and runs it through **3 trained ML models** to predict the most suitable career paths for that candidate — turning raw resume text into actionable career insights.

---

## 🛠️ Tech Stack

- **Flask** — web app framework
- **pandas / numpy** — data processing
- **scikit-learn** — machine learning models
- **matplotlib** — visualizations
- **pdfplumber / pypdf** — PDF resume parsing
- **python-docx** — Word resume parsing

---

## 📋 Requirements

Make sure you have **Python 3.8+** installed, then install the required libraries:

```bash
pip install flask pandas numpy scikit-learn matplotlib pdfplumber pypdf python-docx
```

---

## 🚀 How to Run

### Step 1 — Convert your dataset

```bash
python convert.py
```

### Step 2 — Train the models

```bash
python ml_model.py resumes_ml_ready.csv --save model.pkl
```

### Step 3 — Start the app

```bash
python app.py
```

Then open **[http://localhost:5000](http://localhost:5000)** in your browser. 🎉

---

## 📝 Notes

> ⚠️ **Before running:**
> - `dataset.csv` must include a **`Role`** column (with multiple different job titles) and a **`Resume`** column (with resume text).
> - `model.pkl` must be generated in **Step 2** before running the app in **Step 3**.

---

## 🖼️ Screenshots

> _Add screenshots of your app here, e.g.:_
> ```markdown
> ![Home Page](screenshots/home.png)
> ![Prediction Result](screenshots/result.png)
> ```

---

## 🗺️ Roadmap

- [ ] Support for more resume formats
- [ ] Confidence scores for each predicted career path
- [ ] Downloadable PDF report of recommendations

---

## 📄 License

This project is for educational purposes.

