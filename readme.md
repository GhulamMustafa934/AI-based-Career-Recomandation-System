�

🧭 Career Path Recommender
An ML-powered web app that reads a resume and predicts the best career paths
�
￼ ￼ ￼ ￼ ￼ 


Upload a resume, and let 3 trained machine learning models tell you which career paths fit best.
�


✨ Features · 🛠 Tech Stack · 🚀 Quick Start · 🗺 Roadmap
�

�


✨ What It Does
This app takes a resume — PDF or Word — and runs it through 3 trained ML models to predict the most suitable career paths for that candidate, turning raw resume text into clear, actionable career insights.
�

📤 Upload
🧠 Analyze
🎯 Predict
Drop in a .pdf or .docx resume
Text is parsed & vectorized
3 models vote on the best-fit roles
�

🛠️ Tech Stack
Category
Tools
🌐 Web Framework
Flask
📊 Data Processing
pandas, numpy
🤖 Machine Learning
scikit-learn
📈 Visualization
matplotlib
📄 Resume Parsing
pdfplumber, pypdf, python-docx
📋 Requirements
Make sure you have Python 3.8+ installed, then install the required libraries:
Bash
🚀 How to Run
�
1
Convert your dataset
Bash
�
2
Train the models
Bash
�
3
Start the app
Bash
�

Then open http://localhost:5000 in your browser. 🎉
📝 Notes
⚠️ Before running:
dataset.csv must include a Role column (with multiple different job titles) and a Resume column (with resume text).
model.pkl must be generated in Step 2 before running the app in Step 3.
🖼️ Screenshots
�

Home Page
Prediction Result
�
Load image
�
Load image
�

Add your own screenshots to a screenshots/ folder and update the paths above.
🗺️ Roadmap
[ ] 📁 Support for more resume formats
[ ] 📊 Confidence scores for each predicted career path
[ ] 📄 Downloadable PDF report of recommendations
📄 License
�

This project is for educational purposes.
Made with 🧠 + ☕
�