# RecoFlow

**RecoFlow** is a production-style recommendation system that follows real-world industry architecture: **candidate generation + ranking**, combined with an **end-to-end ML lifecycle** including data pipelines, model training pipelines, API serving, monitoring, and automated retraining.

This project is designed as a **capstone ML engineering project** demonstrating how modern recommendation systems are built, deployed, and maintained in production environments.

The focus of the project is not only on machine learning models, but also on **system design, scalability, and ML lifecycle management (MLOps).**

---

# 🎯 Project Goals

The main objectives of this project are:

• Build a **two-stage recommendation system**

* Stage 1 → Candidate Generation
* Stage 2 → Ranking

• Implement a **reproducible ML pipeline**

• Train and manage **multiple models**

• Expose recommendations via a **production-ready API**

• Implement **monitoring and drift detection**

• Enable **automated model retraining**

---

# 🧠 High-Level Architecture

User interaction data
↓
Data ingestion & validation
↓
Data splitting (temporal user split)
↓
Candidate generation model
↓
Candidate items per user
↓
Ranking feature generation
↓
Ranking model
↓
Top-K recommendations
↓
FastAPI serving layer
↓
Recommendation API
↓
Monitoring & drift detection
↓
Automated retraining pipelines

---

# 🧩 Two-Stage Recommendation Architecture

Modern recommendation systems use a **two-stage architecture** to handle large-scale item catalogs efficiently.

## Stage 1 – Candidate Generation

### Purpose

Retrieve a **small subset of potentially relevant items** for each user from a large catalog.

### Model

Collaborative filtering using **matrix factorization (ALS-style approach)**.

### Output

Top-N candidate items per user.

This stage improves **scalability**, because it avoids scoring every item for every user.

---

## Stage 2 – Ranking

### Purpose

Rank candidate items using **additional behavioral and statistical features**.

### Model

Supervised machine learning model trained on candidate features.

### Example Features

• User interaction statistics
• Item popularity signals
• Candidate model prediction score
• User activity features

### Output

Final **Top-K ranked recommendations** for each user.

---

# ⚙️ ML Pipeline Architecture

RecoFlow includes multiple **training and data pipelines** to simulate a production ML workflow.

## Candidate Training Pipeline

Responsible for:

• Data ingestion
• Data validation
• Temporal data splitting
• Training the candidate generation model

Pipeline file:

```
pipelines/train_candidate_pipeline.py
```

Outputs:

• Candidate model
• User mappings
• Item mappings

---

## Ranking Training Pipeline

Responsible for:

• Generating candidate items
• Building ranking features
• Training the ranking model

Pipeline file:

```
pipelines/train_ranker_pipeline.py
```

Outputs:

• Ranking model used during recommendation

---

# 🌐 API Serving Layer

RecoFlow exposes recommendations through a **FastAPI service**.

The API loads trained models and generates recommendations in real time.

Example endpoint:

```
GET /recommend/{user_id}
```

Example response:

```
{
  "user_id": 10,
  "recommendations": [823, 896, 73, 488, 388]
}
```

The serving layer uses:

• Candidate generation model
• Ranking model
• Feature builder

to compute final recommendations.

---

# 📊 Monitoring

The system includes **data drift monitoring** to detect changes in interaction patterns.

Monitoring compares:

• Training data statistics
• New production data statistics

Example monitored metrics:

• Number of interactions
• Number of users
• Number of items
• Average interactions per user

If the change exceeds a defined threshold, **drift is detected**.

Monitoring code:

```
src/monitoring/drift.py
```

---

# 🔄 Automated Retraining

When drift is detected, the system automatically triggers **model retraining pipelines**.

Retraining workflow:

Drift detection
↓
Candidate model retraining
↓
Ranking model retraining
↓
New models saved for serving

Retraining pipeline:

```
pipelines/retraining_pipeline.py
```

---

# 🛠 Technology Stack

Python

Core libraries:

• pandas
• numpy

Machine learning:

• scikit-learn

API serving:

• FastAPI
• Uvicorn

MLOps components:

• Logging system
• Modular pipelines
• Drift monitoring
• Automated retraining

---

# 📊 Evaluation Strategy

The project evaluates different parts of the system separately.

## Candidate Model

Evaluated using:

• Interaction prediction performance

## Final Recommendation Quality

Evaluated using ranking metrics such as:

• Precision@K
• Recall@K
• NDCG@K

These metrics reflect **real recommendation system evaluation**.

---

# 🧪 Dataset

The system uses the **MovieLens dataset** as a user–item interaction dataset.

This dataset simulates real-world recommendation scenarios with:

• users
• items (movies)
• interaction history

---

# 📂 Project Structure

```
RecoFlow
│
├── data
│   ├── raw
│   └── processed
│
├── models
│
├── pipelines
│   ├── train_candidate_pipeline.py
│   ├── train_ranker_pipeline.py
│   ├── drift_monitoring_pipeline.py
│   └── retraining_pipeline.py
│
├── src
│   ├── ingestion
│   ├── splitting
│   ├── candidate_generation
│   ├── ranking
│   ├── serving
│   ├── monitoring
│   └── common
│
└── main.py
```

---

# 👨‍💻 Author

**Ridhul Krishna P**

This project was built as a **capstone ML engineering project** focusing on:

• recommendation system architecture
• scalable ML system design
• MLOps pipelines
• model serving and monitoring

---

# ⭐ Summary

RecoFlow is not just a recommender model.

It is a **complete recommendation system platform** demonstrating:

• real-world system architecture
• two-stage recommendation design
• ML engineering workflows
• API serving
• monitoring and retraining pipelines

The goal of this project is to showcase how **modern recommendation systems are built and maintained in production environments**.
