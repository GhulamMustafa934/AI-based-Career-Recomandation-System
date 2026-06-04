# Career Path Recommender

An ML web app that reads a resume and predicts the best career paths using 3 trained models.

---

## Requirements

Make sure you have **Python 3.8+** installed, then install the required libraries:

```bash
pip install flask pandas numpy scikit-learn matplotlib pdfplumber pypdf python-docx
```

---

## How to Run

**Step 1 — Convert your dataset**
```bash
python convert.py
```

**Step 2 — Train the models**
```bash
python ml_model.py resumes_ml_ready.csv --save model.pkl
```

**Step 3 — Start the app**
```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Notes

- `dataset.csv` must have a `Role` column with multiple different job titles and a `Resume` column with resume text.
- `model.pkl` must be generated (Step 2) before running the app (Step 3).
