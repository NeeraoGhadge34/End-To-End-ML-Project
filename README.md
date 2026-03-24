# 🎓 Student Performance Prediction (End-to-End ML Project)

## 🚀 Overview

This project demonstrates a **complete end-to-end Machine Learning pipeline** — from data ingestion to deployment on AWS using Docker, ECR, and EC2, along with CI/CD automation.

The application predicts student performance based on features like gender, parental education, and test preparation course.

---

## 🧠 Problem Statement

Predict a student's performance (Maths Score) using demographic and academic input features.

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Flask
* Docker
* AWS (ECR + EC2)
* GitHub Actions (CI/CD)

---

## 🏗️ Project Architecture

Data → Ingestion → Transformation → Training → Model.pkl
↓
Prediction Pipeline → API → Docker → AWS (ECR + EC2)
↓
User

---

## 🔧 Features

* Modular ML pipeline (ingestion, transformation, training)
* Hyperparameter tuning using GridSearchCV
* Multiple model comparison
* Custom exception handling & logging
* Prediction pipeline for real-time inference
* Flask-based web application
* Docker containerization
* CI/CD pipeline using GitHub Actions
* Deployment on AWS using ECR and EC2

---

## 📂 Project Structure

<img width="1919" height="1020" alt="Screenshot 2026-03-24 165912" src="https://github.com/user-attachments/assets/03a4d711-92a0-4e56-8cdb-bf46ba38782e" />


## 🌐 Deployment

* Dockerized the application
* Pushed image to AWS ECR
* Deployed container on AWS EC2
* Automated CI/CD using GitHub Actions with security keys

---

## 📸 Demo
<img width="945" height="732" alt="Screenshot 2026-03-19 044638" src="https://github.com/user-attachments/assets/1bc05c3e-e887-46d2-852e-abc3787615fc" />

<img width="1125" height="999" alt="image" src="https://github.com/user-attachments/assets/d3b0e123-6576-414a-b86b-d6975ee35856" />


## ▶️ How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/NeeraoGhadge34/End-To-End-ML-Project.git
cd End-To-End-ML-Project
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the app

```
python app.py
```

---


## 🤝 Connect

If you found this project interesting, feel free to connect or reach out!

---

## ⭐ Give a Star

If you like this project, consider giving it a ⭐ on GitHub!
