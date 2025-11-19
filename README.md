🚨 Intrusion Detection System (IDS) Using Machine Learning

A production-grade implementation of a Machine Learning–based Intrusion Detection System designed to identify malicious network activity using statistical and behavioral analysis.

📘 Overview

This project demonstrates the end-to-end workflow of building an IDS using modern machine learning techniques.
It includes data preprocessing, feature engineering, model selection, backend API for inference, and a frontend interface for real-time monitoring.

The goal is to provide a modular, scalable, and reproducible framework suitable for academic research, cybersecurity demonstrations, and enterprise-level experimentation.

🏗️ Repository Structure
Intrusion-detection-system-using-machine-learning
│
├── Backend/
│   ├── data/               # Dataset files (training & testing)
│   ├── models/             # Saved ML models
│   ├── src/                # Preprocessing, training, evaluation scripts
│   ├── API/                # Flask/FastAPI backend for serving predictions
│   └── requirements.txt    # Python dependencies
│
├── Frontend/
│   ├── public/             # Static assets
│   ├── src/                # React application source code
│   └── package.json        # Frontend dependencies
│
└── README.md

🛠️ Tech Stack
Backend & Machine Learning

Python 3.x

NumPy, Pandas

Scikit-learn

Flask / FastAPI

Joblib for model serialization

Frontend

React.js

JavaScript/TypeScript

Axios for API communication

Tools

Git & GitHub

Jupyter Notebooks

VS Code / PyCharm

📊 Machine Learning Workflow
✔️ 1. Data Preparation

Handling missing values

Normalization & scaling

Encoding categorical features

Train–test splits

✔️ 2. Model Development

Algorithms typically include:

Random Forest Classifier

Logistic Regression

Decision Trees

Naive Bayes

Hyperparameter tuning using GridSearch / manual optimization

✔️ 3. Evaluation

Common metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

✔️ 4. Deployment

REST API endpoint for real-time predictions

Frontend interface to visualize alerts & predictions

🚀 Getting Started
Prerequisites

Install the following:

Python 3.x
Node.js & npm
Git

Backend Setup
cd Backend
pip install -r requirements.txt
python src/train_model.py     # Train or load model
python API/app.py             # Start backend server

Frontend Setup
cd Frontend
npm install
npm start


Frontend usually runs at:
➡️ http://localhost:3000
Backend usually runs at:
➡️ http://localhost:5000

🖥️ Using the System

Once both servers are running:

Upload or stream network traffic data

Monitor intrusion predictions in real-time

View classification results

Visualize alerts and network behavior

Extend the system by adding new models or features

📚 Future Enhancements

Integration with deep learning architectures (LSTM/Autoencoders)

Real-time packet sniffing and streaming predictions

Auto-retraining pipelines

Docker & Kubernetes deployment

Cloud-based dashboards

🤝 Contributing

Contributions are welcome!
Please follow these steps:

Fork the repository

Create a feature branch

Commit your changes

Open a Pull Request


✉️ Contact

Author: Vedang Rajoriya
🔗 GitHub: @vedangrajoriya

For queries, collaborations, or improvements—feel free to open an issue.
