#!/usr/bin/env python3
"""
app.py — Career Path Recommender Web UI
-----------------------------------------
A Flask web app that lets users upload a resume (PDF/DOCX/TXT)
or paste resume text, then shows ML career path predictions
from all 3 trained models.

Usage:
  pip install flask
  python app.py

Then open: http://localhost:5000
"""

import os
import sys
import json
import pickle
import warnings
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore")

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np

# ── Import shared ML functions from ml_model.py ──
sys.path.insert(0, str(Path(__file__).parent))
from ml_model import (
    prepare_features, encode_categoricals, build_feature_matrix,
    predict_career_path,
    NUMERIC_FEATURES, TEXT_FEATURES, CATEGORICAL_FEATURES,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# ── Global model bundle ──
MODEL_BUNDLE = None
MODEL_PATH   = "model.pkl"
CSV_PATH     = "resumes_ml_ready.csv"


# ─────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Career Path Recommender — AI Resume Analyzer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh; color: #fff;
  }
  .hero {
    text-align: center; padding: 48px 20px 32px;
  }
  .hero h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px; }
  .hero h1 span { color: #7c83fd; }
  .hero p  { margin-top: 10px; color: #a0aec0; font-size: 1.05rem; }

  .card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 28px;
    backdrop-filter: blur(10px);
  }

  .container { max-width: 900px; margin: 0 auto; padding: 0 20px 60px; }

  .tabs { display: flex; gap: 8px; margin-bottom: 20px; }
  .tab-btn {
    flex: 1; padding: 12px; border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px; background: rgba(255,255,255,0.05);
    color: #a0aec0; cursor: pointer; font-size: 0.95rem; transition: all 0.2s;
  }
  .tab-btn.active { background: #7c83fd; color: #fff; border-color: #7c83fd; }

  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  textarea {
    width: 100%; height: 200px; padding: 14px;
    background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px; color: #fff; font-size: 0.9rem; resize: vertical;
  }
  textarea:focus { outline: none; border-color: #7c83fd; }

  .upload-area {
    border: 2px dashed rgba(124,131,253,0.5); border-radius: 12px;
    padding: 40px; text-align: center; cursor: pointer;
    transition: all 0.2s; color: #a0aec0;
  }
  .upload-area:hover, .upload-area.dragover {
    border-color: #7c83fd; background: rgba(124,131,253,0.08);
  }
  .upload-area .icon { font-size: 3rem; margin-bottom: 12px; }
  .upload-area input { display: none; }

  .btn {
    width: 100%; padding: 14px; margin-top: 16px;
    background: linear-gradient(135deg, #7c83fd, #5a63f5);
    border: none; border-radius: 10px; color: #fff;
    font-size: 1rem; font-weight: 700; cursor: pointer; transition: all 0.2s;
  }
  .btn:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(124,131,253,0.4); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* Results */
  #results { margin-top: 28px; display: none; }

  .model-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin-bottom: 16px;
    background: linear-gradient(135deg, #7c83fd, #5a63f5);
  }

  .prediction-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 16px;
  }
  .prediction-card.rank-1 { border-color: #ffd700; background: rgba(255,215,0,0.06); }
  .rank-icon { font-size: 1.8rem; min-width: 40px; text-align: center; }
  .pred-info { flex: 1; }
  .pred-title { font-size: 1.05rem; font-weight: 700; }
  .pred-bar-wrap { margin-top: 6px; background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; }
  .pred-bar { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #7c83fd, #5a63f5); }
  .pred-bar.high  { background: linear-gradient(90deg, #48bb78, #38a169); }
  .pred-bar.medium{ background: linear-gradient(90deg, #ed8936, #dd6b20); }
  .pred-conf { font-size: 1.1rem; font-weight: 800; min-width: 60px; text-align: right; }
  .fit-badge {
    padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;
  }
  .fit-High   { background: rgba(72,187,120,0.2); color: #48bb78; }
  .fit-Medium { background: rgba(237,137,54,0.2); color: #ed8936; }
  .fit-Low    { background: rgba(160,174,192,0.2); color: #a0aec0; }

  /* Comparison table */
  .comparison { margin-top: 24px; }
  .comparison h3 { font-size: 1rem; color: #a0aec0; margin-bottom: 12px; }
  .cmp-row {
    display: flex; align-items: center; gap: 12px;
    background: rgba(255,255,255,0.04); border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.06);
  }
  .cmp-row.winner { border-color: #ffd700; }
  .cmp-name { flex: 1; font-weight: 600; font-size: 0.9rem; }
  .cmp-acc  { font-size: 0.9rem; color: #a0aec0; }
  .cmp-pred { font-weight: 700; color: #7c83fd; font-size: 0.9rem; }

  /* Status card */
  .status-card {
    display: flex; align-items: center; gap: 12px; padding: 14px 18px;
    border-radius: 10px; margin-bottom: 20px; font-size: 0.9rem;
  }
  .status-ok    { background: rgba(72,187,120,0.15); border: 1px solid rgba(72,187,120,0.3); color: #48bb78; }
  .status-warn  { background: rgba(237,137,54,0.15); border: 1px solid rgba(237,137,54,0.3); color: #ed8936; }
  .status-error { background: rgba(245,101,101,0.15); border: 1px solid rgba(245,101,101,0.3); color: #fc8181; }

  .spinner {
    display: inline-block; width: 20px; height: 20px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff; border-radius: 50%;
    animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  label { display: block; font-size: 0.85rem; color: #a0aec0; margin-bottom: 6px; }
</style>
</head>
<body>

<div class="hero">
  <h1>Career Path <span>Recommender</span></h1>
  <p>Upload your resume — our ML models predict the best career path for you</p>
</div>

<div class="container">

  <!-- Model status -->
  <div id="model-status"></div>

  <div class="card">
    <!-- Tabs -->
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('text')">📝 Paste Resume Text</button>
      <button class="tab-btn"        onclick="switchTab('file')">📁 Upload File (PDF/DOCX/TXT)</button>
    </div>

    <!-- Tab: Text -->
    <div id="tab-text" class="tab-pane active">
      <label>Paste your resume text below:</label>
      <textarea id="resume-text" placeholder="Paste your full resume here...
Example:
John Doe | Python Developer
Skills: Python, Django, PostgreSQL, Docker, Git, AWS
Experience: 3 years at TechCorp as Backend Developer
Education: Bachelor in Computer Science
Certifications: AWS Solutions Architect"></textarea>
      <button class="btn" id="btn-text" onclick="analyzeText()">🔍 Analyze Resume</button>
    </div>

    <!-- Tab: File -->
    <div id="tab-file" class="tab-pane">
      <div class="upload-area" id="drop-zone" onclick="document.getElementById('file-input').click()"
           ondragover="handleDrag(event,true)" ondragleave="handleDrag(event,false)" ondrop="handleDrop(event)">
        <div class="icon">📄</div>
        <p><strong>Click to upload</strong> or drag & drop</p>
        <p style="font-size:0.85rem;margin-top:6px;color:#718096">PDF, DOCX, or TXT — max 10 MB</p>
        <input type="file" id="file-input" accept=".pdf,.docx,.doc,.txt" onchange="fileSelected(this)">
      </div>
      <p id="file-name" style="margin-top:10px;font-size:0.9rem;color:#a0aec0;text-align:center;"></p>
      <button class="btn" id="btn-file" onclick="analyzeFile()" disabled>🔍 Analyze Resume</button>
    </div>
  </div>

  <!-- Results -->
  <div id="results">
    <div class="card" style="margin-top:0;">

      <div id="best-model-badge"></div>
      <h2 style="font-size:1.3rem;margin-bottom:16px;">🎯 Career Path Predictions</h2>
      <div id="predictions-list"></div>

      <div class="comparison" id="comparison-section">
        <h3>📊 All Models Comparison</h3>
        <div id="comparison-list"></div>
      </div>

    </div>
  </div>

</div>

<script>
  // ── Tab switching ──
  function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', (tab==='text'?i===0:i===1)));
    document.getElementById('tab-text').classList.toggle('active', tab==='text');
    document.getElementById('tab-file').classList.toggle('active', tab==='file');
  }

  // ── File handling ──
  function handleDrag(e, over) { e.preventDefault(); document.getElementById('drop-zone').classList.toggle('dragover', over); }
  function handleDrop(e) {
    e.preventDefault(); handleDrag(e, false);
    const f = e.dataTransfer.files[0]; if (f) { document.getElementById('file-input').files = e.dataTransfer.files; fileSelected(document.getElementById('file-input')); }
  }
  function fileSelected(input) {
    const f = input.files[0];
    document.getElementById('file-name').textContent = f ? `Selected: ${f.name}` : '';
    document.getElementById('btn-file').disabled = !f;
  }

  // ── Check model status on load ──
  async function checkStatus() {
    const res = await fetch('/status');
    const data = await res.json();
    const el = document.getElementById('model-status');
    if (data.ready) {
      el.innerHTML = `<div class="status-card status-ok">✅ Model loaded — trained on ${data.classes} career paths using ${data.best_model}</div>`;
    } else {
      el.innerHTML = `<div class="status-card status-warn">⚠️ ${data.message} — <a href="/train" style="color:#ed8936;text-decoration:underline;">Click here to train</a></div>`;
    }
  }
  checkStatus();

  // ── Analyze text ──
  async function analyzeText() {
    const text = document.getElementById('resume-text').value.trim();
    if (!text) { alert('Please paste resume text first.'); return; }
    const btn = document.getElementById('btn-text');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing...';
    try {
      const res  = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text}) });
      const data = await res.json();
      showResults(data);
    } catch(e) { alert('Error: ' + e.message); }
    btn.disabled = false;
    btn.innerHTML = '🔍 Analyze Resume';
  }

  // ── Analyze file ──
  async function analyzeFile() {
    const file = document.getElementById('file-input').files[0];
    if (!file) { alert('Please select a file first.'); return; }
    const btn = document.getElementById('btn-file');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing...';
    const form = new FormData();
    form.append('file', file);
    try {
      const res  = await fetch('/predict_file', { method:'POST', body: form });
      const data = await res.json();
      showResults(data);
    } catch(e) { alert('Error: ' + e.message); }
    btn.disabled = false;
    btn.innerHTML = '🔍 Analyze Resume';
  }

  // ── Render results ──
  function showResults(data) {
    if (data.error) { alert('Error: ' + data.error); return; }

    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Best model badge
    document.getElementById('best-model-badge').innerHTML =
      `<span class="model-badge">🏆 Best Model: ${data.best_model}</span>`;

    // Predictions
    const icons = ['🥇','🥈','🥉','4️⃣','5️⃣'];
    const list  = document.getElementById('predictions-list');
    list.innerHTML = data.predictions.map((p, i) => `
      <div class="prediction-card ${i===0?'rank-1':''}">
        <div class="rank-icon">${icons[i]||''}</div>
        <div class="pred-info">
          <div class="pred-title">${p.career_path}</div>
          <div class="pred-bar-wrap">
            <div class="pred-bar ${p.fit.toLowerCase()}" style="width:${p.confidence}%"></div>
          </div>
        </div>
        <span class="fit-badge fit-${p.fit}">${p.fit} fit</span>
        <div class="pred-conf">${p.confidence}%</div>
      </div>
    `).join('');

    // Comparison
    const cmpList = document.getElementById('comparison-list');
    cmpList.innerHTML = data.all_models.map(m => `
      <div class="cmp-row ${m.is_best?'winner':''}">
        <div class="cmp-name">${m.is_best?'🏆 ':''}${m.model}</div>
        <div class="cmp-pred">→ ${m.top_prediction}</div>
        <div class="cmp-acc">${m.confidence}% confidence</div>
      </div>
    `).join('');
  }
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def load_bundle():
    global MODEL_BUNDLE
    if os.path.isfile(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            MODEL_BUNDLE = pickle.load(f)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠  No model file found at {MODEL_PATH}")
        print(f"   Run: python ml_model.py {CSV_PATH} --save {MODEL_PATH}")


def text_to_row(text: str) -> dict:
    """Convert raw resume text into the feature dict ml_model expects."""
    import re
    text_lower = text.lower()

    TECH = [
        "python","java","javascript","typescript","c++","c#","ruby","go","rust",
        "swift","kotlin","scala","r","php","dart","html","css","react","vue",
        "angular","node.js","django","flask","fastapi","spring","postgresql",
        "mysql","mongodb","redis","docker","kubernetes","aws","azure","gcp",
        "tensorflow","pytorch","scikit-learn","pandas","numpy","sql","git",
        "machine learning","deep learning","nlp","computer vision","linux",
        "tableau","power bi","excel","flutter","android","ios","ci/cd",
    ]
    SOFT = [
        "leadership","communication","teamwork","problem solving",
        "management","analytical","creative","organized",
    ]

    tech = [s for s in TECH if s in text_lower]
    soft = [s for s in SOFT if s in text_lower]

    years_found = re.findall(r"\b(19|20)\d{2}\b", text)
    years_found = [int(y) for y in years_found if 1970 <= int(y) <= 2026]
    exp = max(0, min(max(years_found) - min(years_found), 40)) if len(years_found) >= 2 else 0

    edu = "unknown"
    if any(x in text_lower for x in ["ph.d","phd","doctorate"]):
        edu = "phd"
    elif any(x in text_lower for x in ["master","mba","m.sc"]):
        edu = "master"
    elif any(x in text_lower for x in ["bachelor","b.sc","b.tech"]):
        edu = "bachelor"

    # Clean resume text the same way convert2.py does for TF-IDF compatibility
    resume_clean = re.sub(r"[^a-z0-9\s]", " ", text_lower)
    resume_clean = re.sub(r"\s+", " ", resume_clean).strip()[:2000]

    return {
        # Core numeric features expected by NUMERIC_FEATURES in ml_model.py
        "total_experience_years": exp,
        "num_technical_skills":   len(tech),
        "num_soft_skills":        len(soft),
        "hired":                  0,           # unknown at inference time
        "resume_length":          len(text.split()),
        # Categorical features expected by CATEGORICAL_FEATURES
        "experience_level": "senior" if exp>6 else "mid level" if exp>3 else "junior" if exp>1 else "entry level",
        "education_level":  edu,
        # Text features expected by TEXT_FEATURES — space-separated to match TF-IDF training
        "technical_skills": " ".join(tech),
        "resume_text":      resume_clean,
    }


def run_all_models(row: dict):
    """Run prediction on all 3 trained models and return combined result."""
    bundle  = MODEL_BUNDLE
    best_name_key = bundle.get("best_name", "Best Model")
    trained = bundle.get("all_models", {best_name_key: bundle["best_model"]})
    label_enc    = bundle["label_enc"]
    cat_encoders = bundle["cat_encoders"]
    tfidf_models = bundle["tfidf_models"]
    scaler       = bundle["scaler"]
    feature_names= bundle["feature_names"]
    best_name    = bundle["best_name"]

    best_preds = predict_career_path(
        row, bundle["best_model"], best_name,
        label_enc, cat_encoders, tfidf_models, scaler, feature_names, top_n=5,
    )

    all_models_result = []
    for model_name, clf in trained.items():
        preds = predict_career_path(
            row, clf, model_name,
            label_enc, cat_encoders, tfidf_models, scaler, feature_names, top_n=1,
        )
        if preds:
            all_models_result.append({
                "model":          model_name,
                "top_prediction": preds[0]["career_path"],
                "confidence":     preds[0]["confidence"],
                "is_best":        model_name == best_name,
            })

    return best_preds, all_models_result, best_name


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/status")
def status():
    if MODEL_BUNDLE:
        return jsonify({
            "ready":      True,
            "best_model": MODEL_BUNDLE.get("best_name", "Unknown"),
            "classes":    len(MODEL_BUNDLE["label_enc"].classes_),
        })
    return jsonify({
        "ready":   False,
        "message": f"Model not trained yet. Run: python ml_model.py {CSV_PATH} --save {MODEL_PATH}",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_BUNDLE:
        return jsonify({"error": f"Model not loaded. Run: python ml_model.py {CSV_PATH} --save {MODEL_PATH}"}), 400

    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    row = text_to_row(text)
    best_preds, all_models_result, best_name = run_all_models(row)

    return jsonify({
        "predictions": best_preds,
        "all_models":  all_models_result,
        "best_model":  best_name,
    })


@app.route("/predict_file", methods=["POST"])
def predict_file():
    if not MODEL_BUNDLE:
        return jsonify({"error": f"Model not loaded. Run: python ml_model.py {CSV_PATH} --save {MODEL_PATH}"}), 400

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".doc", ".txt"):
        return jsonify({"error": "Only PDF, DOCX, and TXT files are supported"}), 400

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Extract text
        if suffix == ".txt":
            text = Path(tmp_path).read_text(encoding="utf-8", errors="ignore")
        elif suffix in (".docx", ".doc"):
            from docx import Document
            doc  = Document(tmp_path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif suffix == ".pdf":
            try:
                import pdfplumber
                with pdfplumber.open(tmp_path) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            except Exception:
                from pypdf import PdfReader
                reader = PdfReader(tmp_path)
                text   = "\n".join(p.extract_text() or "" for p in reader.pages)
    finally:
        os.unlink(tmp_path)

    if not text.strip():
        return jsonify({"error": "Could not extract text from file"}), 400

    row = text_to_row(text)
    best_preds, all_models_result, best_name = run_all_models(row)

    return jsonify({
        "predictions": best_preds,
        "all_models":  all_models_result,
        "best_model":  best_name,
    })


@app.route("/train")
def train_route():
    return jsonify({
        "message": f"To train the model run: python ml_model.py {CSV_PATH} --save {MODEL_PATH}",
        "then":    "Restart app.py after training completes."
    })


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("   🌐  Career Path Recommender — Web UI")
    print("═"*55)
    load_bundle()
    print("\n  Open browser at: http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
