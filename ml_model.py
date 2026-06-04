#!/usr/bin/env python3
"""
ml_model.py — Career Role Predictor
-------------------------------------
Trains and compares 3 ML models on resume data:
  1. Decision Tree
  2. Random Forest
  3. Logistic Regression

Data split: 70% train | 15% validation | 15% test  (from a 70% sample)

Usage:
  python ml_model.py resumes_ml_ready.csv
  python ml_model.py resumes_ml_ready.csv --save model.pkl
  python ml_model.py resumes_ml_ready.csv --row 0
  python ml_model.py --load model.pkl resumes_ml_ready.csv
"""

import sys, os, argparse, warnings, pickle
from datetime import datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def clr_print(msg, code="0"):
    print(f"\033[{code}m{msg}\033[0m")

NUMERIC_FEATURES     = ["total_experience_years","num_technical_skills","num_soft_skills","hired","resume_length"]
TEXT_FEATURES        = ["technical_skills","resume_text"]
CATEGORICAL_FEATURES = ["experience_level","education_level"]
EXPERIENCE_ORDER     = ["entry level","junior","mid level","senior","executive"]
EDUCATION_ORDER      = ["unknown","associate","bachelor","master","phd"]


def prepare_features(df):
    df = df.copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0) if col in df.columns else 0
    for col in TEXT_FEATURES:
        df[col] = df[col].fillna("").astype(str) if col in df.columns else ""
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str).str.lower() if col in df.columns else "unknown"
    return df


def encode_categoricals(df, encoders=None, fit=True):
    encoders = encoders or {}
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        order = EXPERIENCE_ORDER if col == "experience_level" else EDUCATION_ORDER
        df[col] = df[col].map({v: i for i, v in enumerate(order)}).fillna(0).astype(int)
    return df, encoders


def build_feature_matrix(df, tfidf_models=None, fit=True):
    tfidf_models = tfidf_models or {}
    parts, feature_names = [], []

    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    parts.append(df[num_cols].values.astype(float))
    feature_names.extend(num_cols)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    parts.append(df[cat_cols].values.astype(float))
    feature_names.extend(cat_cols)

    for col in TEXT_FEATURES:
        texts = df[col].fillna("").astype(str).tolist()
        if not any(t.strip() for t in texts):
            parts.append(np.zeros((len(df), 1)))
            feature_names.append(f"{col}__empty")
            continue
        if fit:
            tfidf = TfidfVectorizer(max_features=50 if col=="resume_text" else 30,
                                     token_pattern=r"[a-z][a-z0-9+#.\-]{1,}",
                                     ngram_range=(1,2), sublinear_tf=True)
            try:
                mat = tfidf.fit_transform(texts).toarray()
                tfidf_models[col] = tfidf
            except ValueError:
                parts.append(np.zeros((len(df),1)))
                feature_names.append(f"{col}__empty")
                continue
        else:
            tfidf = tfidf_models.get(col)
            mat   = tfidf.transform(texts).toarray() if tfidf else np.zeros((len(df),1))
        parts.append(mat)
        names = tfidf_models[col].get_feature_names_out() if col in tfidf_models else ["?"]
        feature_names.extend([f"{col}__{t}" for t in names])

    return np.hstack(parts), feature_names, tfidf_models


def train_all_models(df):
    df = df.copy()
    df["career_path"] = df["career_category"].fillna("Unknown").astype(str).str.strip().str.title()

    label_counts = df["career_path"].value_counts()
    clr_print(f"\n  📊  Label Distribution (full dataset: {len(df)} rows):", "36")
    for path, count in label_counts.items():
        bar = "█" * min(int(count / max(label_counts.max()/30, 1)), 30)
        print(f"      {path:<35} {count:>5}  {bar}")

    valid   = label_counts[label_counts >= 2].index
    dropped = label_counts[label_counts < 2].index.tolist()
    if dropped:
        clr_print(f"\n  ⚠   Dropping rare classes: {dropped}", "33")
    df = df[df["career_path"].isin(valid)].copy()

    if len(df) < 10:
        raise ValueError("Need at least 10 rows. Run convert2.py first.")
    if len(label_counts[label_counts >= 2]) < 2:
        raise ValueError(
            "\n  ❌  Only one unique Role found in your data!\n"
            "  Your 'Role' column must have multiple different job titles.\n"
            "  Check dataset.csv and make sure it has diverse roles like:\n"
            "  'Data Scientist', 'Backend Developer', 'Product Manager', etc.\n"
        )

    df = df.sample(frac=0.70, random_state=42).reset_index(drop=True)
    clr_print(f"\n  ✂   Using 70% of dataset → {len(df)} rows", "36")

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df["career_path"])
    df = prepare_features(df)
    df, cat_encoders = encode_categoricals(df, fit=True)
    X, feature_names, tfidf_models = build_feature_matrix(df, fit=True)

    clr_print(f"\n  🔢  Feature matrix : {X.shape[0]} × {X.shape[1]}", "36")
    clr_print(f"  🎯  Classes ({len(label_enc.classes_)}): {list(label_enc.classes_)}", "36")

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15/0.85, random_state=42, stratify=y_tv)

    total = len(y)
    clr_print(f"\n  📐  Split Summary ({total} rows):", "36")
    clr_print(f"      🟢 Train      : {len(y_train):>5}  ({len(y_train)/total*100:.1f}%)", "32")
    clr_print(f"      🟡 Validation : {len(y_val):>5}  ({len(y_val)/total*100:.1f}%)", "33")
    clr_print(f"      🔴 Test       : {len(y_test):>5}  ({len(y_test)/total*100:.1f}%)", "31")

    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_train)
    X_va_sc    = scaler.transform(X_val)
    X_te_sc    = scaler.transform(X_test)
    X_fu_sc    = scaler.transform(X)

    n_splits = min(5, int(pd.Series(y_train).value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models_def = {
        "Decision Tree":      DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
        "Random Forest":      RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "Logistic Regression":LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42),
    }

    results, trained = {}, {}
    clr_print("\n" + "═"*65, "1;36")
    clr_print("   📊  COMPARATIVE MODEL ANALYSIS  (70 / 15 / 15 split)", "1;37")
    clr_print("═"*65, "1;36")

    for name, clf in models_def.items():
        clr_print(f"\n  🔧  Training: {name}...", "35")
        Xtr, Xval, Xte, Xfu = (X_tr_sc, X_va_sc, X_te_sc, X_fu_sc) if name=="Logistic Regression" else (X_train, X_val, X_test, X)

        clf.fit(Xtr, y_train)
        trained[name] = clf

        val_acc  = accuracy_score(y_val,  clf.predict(Xval))
        y_pred   = clf.predict(Xte)
        test_acc = accuracy_score(y_test, y_pred)
        cv_s     = cross_val_score(clf, Xfu, y, cv=cv, scoring="accuracy")

        results[name] = {"val_accuracy": val_acc, "test_accuracy": test_acc,
                         "cv_mean": cv_s.mean(), "cv_std": cv_s.std(), "y_pred": y_pred}

        clr_print(f"  🟡  Validation Accuracy : {val_acc:.2%}", "33")
        clr_print(f"  🔴  Test Accuracy       : {test_acc:.2%}", "32")
        clr_print(f"  📊  CV Accuracy         : {cv_s.mean():.2%} ± {cv_s.std():.2%}", "32")
        print(classification_report(y_test, y_pred, target_names=label_enc.classes_, zero_division=0))

    best_name  = max(results, key=lambda n: results[n]["val_accuracy"])
    best_model = trained[best_name]

    clr_print("\n" + "═"*65, "1;33")
    clr_print(f"   🏆  BEST MODEL : {best_name}", "1;33")
    clr_print(f"   🟡  Val  : {results[best_name]['val_accuracy']:.2%}   🔴  Test : {results[best_name]['test_accuracy']:.2%}   📊  CV : {results[best_name]['cv_mean']:.2%}", "33")
    clr_print("═"*65, "1;33")

    clr_print(f"\n  {'Model':<25} {'Val Acc':>9} {'Test Acc':>10} {'CV Acc':>10} {'CV Std':>8}", "1;36")
    clr_print("  " + "─"*68, "90")
    for n, r in sorted(results.items(), key=lambda x: -x[1]["val_accuracy"]):
        trophy = " 🏆" if n == best_name else ""
        print(f"  {n:<25} {r['val_accuracy']:>8.2%} {r['test_accuracy']:>9.2%} {r['cv_mean']:>9.2%} {r['cv_std']:>7.2%}{trophy}")

    save_comparison_chart(results, "model_comparison.png")
    if "Random Forest" in trained:
        save_feature_importance(trained["Random Forest"], feature_names, "feature_importance.png")
    save_confusion_matrix(y_test, results[best_name]["y_pred"], label_enc.classes_, best_name, "confusion_matrix.png")

    return best_model, best_name, trained, results, label_enc, cat_encoders, tfidf_models, scaler, feature_names


def predict_career_path(row, model, model_name, label_enc, cat_encoders, tfidf_models, scaler, feature_names, top_n=5):
    df_row = pd.DataFrame([row])
    df_row = prepare_features(df_row)
    df_row, _ = encode_categoricals(df_row, encoders=cat_encoders, fit=False)
    X_row, _, _ = build_feature_matrix(df_row, tfidf_models=tfidf_models, fit=False)
    if model_name == "Logistic Regression":
        X_row = scaler.transform(X_row)
    proba = model.predict_proba(X_row)[0]
    top_idx = np.argsort(proba)[::-1][:top_n]
    return [{"rank": r+1, "career_path": label_enc.classes_[i],
              "confidence": round(float(proba[i])*100, 1),
              "fit": "High" if proba[i]>=0.5 else "Medium" if proba[i]>=0.2 else "Low"}
            for r, i in enumerate(top_idx) if proba[i] >= 0.001]


def save_comparison_chart(results, out_path):
    names   = list(results.keys())
    val_acc = [results[n]["val_accuracy"]*100  for n in names]
    tst_acc = [results[n]["test_accuracy"]*100 for n in names]
    cv_acc  = [results[n]["cv_mean"]*100       for n in names]
    cv_std  = [results[n]["cv_std"]*100        for n in names]
    x, w    = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(12,6))
    b1 = ax.bar(x-w, val_acc, w, label="Validation (15%)", color="#F4A261", edgecolor="white")
    b2 = ax.bar(x,   tst_acc, w, label="Test (15%)",        color="#2E86AB", edgecolor="white")
    b3 = ax.bar(x+w, cv_acc,  w, label="CV Accuracy",        color="#A8DADC", edgecolor="white",
                yerr=cv_std, capsize=5, error_kw={"elinewidth":1.5,"ecolor":"#555"})
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison — Career Role Predictor (70/15/15)", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11); ax.set_ylim(0,120); ax.legend(fontsize=10)
    for b in [b1,b2,b3]: ax.bar_label(b, fmt="%.1f%%", padding=3, fontsize=9)
    best_idx = int(np.argmax(val_acc))
    ax.get_xticklabels()[best_idx].set_color("#E63946"); ax.get_xticklabels()[best_idx].set_fontweight("bold")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    clr_print(f"  📊  Chart saved → {out_path}", "32")


def save_feature_importance(model, feature_names, out_path):
    imp = model.feature_importances_
    top_n = min(25, len(feature_names))
    top_idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(12,8))
    colors = ["#2E86AB" if imp[i]>=0.03 else "#A8DADC" for i in top_idx]
    ax.barh([feature_names[i] for i in reversed(top_idx)], [imp[i] for i in reversed(top_idx)],
            color=list(reversed(colors)), edgecolor="white")
    ax.set_xlabel("Importance Score"); ax.set_title("Top Feature Importances — Random Forest", fontweight="bold")
    ax.axvline(0.03, color="red", linestyle="--", linewidth=1, label="3% threshold"); ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    clr_print(f"  📈  Feature importance saved → {out_path}", "32")


def save_confusion_matrix(y_true, y_pred, class_names, model_name, out_path):
    cm = confusion_matrix(y_true, y_pred)
    n  = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n*0.6), max(6, n*0.5)))
    cmap = LinearSegmentedColormap.from_list("cm", ["#ffffff","#2E86AB"])
    im = ax.imshow(cm, cmap=cmap); plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    lbl = [c[:16]+".." if len(c)>18 else c for c in class_names]
    ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=max(6,10-n//5))
    ax.set_yticklabels(lbl, fontsize=max(6,10-n//5))
    thresh = cm.max()/2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=max(5,9-n//5), color="white" if cm[i,j]>thresh else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    clr_print(f"  🔲  Confusion matrix saved → {out_path}", "32")


def save_bundle(path, best_model, best_name, trained, label_enc, cat_encoders, tfidf_models, scaler, feature_names):
    with open(path, "wb") as f:
        pickle.dump({"best_model":best_model,"best_name":best_name,"all_models":trained,
                     "label_enc":label_enc,"cat_encoders":cat_encoders,"tfidf_models":tfidf_models,
                     "scaler":scaler,"feature_names":feature_names,"saved_at":datetime.now().isoformat()}, f)
    clr_print(f"\n  💾  Model saved → {path}", "1;32")


def load_bundle(path):
    with open(path,"rb") as f: b = pickle.load(f)
    clr_print(f"\n  📦  Loaded ← {path}  |  Best: {b['best_name']}  |  Classes: {list(b['label_enc'].classes_)}", "1;32")
    return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?")
    parser.add_argument("--save", default="")
    parser.add_argument("--load", default="")
    parser.add_argument("--row", type=int, default=-1)
    args = parser.parse_args()

    clr_print("\n" + "═"*62, "1;36")
    clr_print("   🌳  CAREER ROLE PREDICTOR — ML Comparative Analysis", "1;37")
    clr_print("   Decision Tree  |  Random Forest  |  Logistic Regression", "37")
    clr_print("═"*62, "1;36")

    if not args.csv_path: parser.print_help(); sys.exit(1)
    if not os.path.isfile(args.csv_path): clr_print(f"\n  ❌  Not found: {args.csv_path}\n","31"); sys.exit(1)

    clr_print(f"  📂  Loading: {args.csv_path}", "36")
    df = pd.read_csv(args.csv_path); df.columns = df.columns.str.strip()
    clr_print(f"  ✅  {len(df)} candidates | {len(df.columns)} columns", "32")
    clr_print(f"  🏋   Training all 3 models...", "36")

    if args.load and os.path.isfile(args.load):
        b = load_bundle(args.load)
        best_model, best_name, trained = b["best_model"], b["best_name"], b["all_models"]
        label_enc, cat_encoders, tfidf_models, scaler, feature_names = \
            b["label_enc"], b["cat_encoders"], b["tfidf_models"], b["scaler"], b["feature_names"]
    else:
        best_model, best_name, trained, results, label_enc, cat_encoders, tfidf_models, scaler, feature_names = train_all_models(df)
        if args.save:
            save_bundle(args.save, best_model, best_name, trained, label_enc, cat_encoders, tfidf_models, scaler, feature_names)

    if args.row >= 0:
        row = df.iloc[args.row].to_dict()
        preds = predict_career_path(row, best_model, best_name, label_enc, cat_encoders, tfidf_models, scaler, feature_names)
        clr_print(f"\n  📋  Top Predictions ({best_name}):", "1;36")
        for p in preds:
            print(f"  #{p['rank']}  {p['career_path']:<35} {p['confidence']:>5.1f}%  {'█'*int(p['confidence']/5)}  [{p['fit']}]")

    clr_print("\n  ✅  Done!\n", "1;32")


if __name__ == "__main__":
    main()
