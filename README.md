# RecoFlow
RecoFlow is a production-oriented recommendation system that follows real-world industry architecture: candidate generation + ranking, wrapped with a complete MLOps lifecycle including training pipelines, experiment tracking, model registry, deployment, monitoring and automated retraining.

This project is designed as a capstone to demonstrate how modern recommendation systems are built, deployed and maintained in real production environments.

This project focuses on system design and lifecycle management, not only on building a machine learning model.

Specifically, the goals are:

 -> Build a two-stage recommendation system

  -> Stage 1: candidate generation

  -> Stage 2: ranking

-> Implement a reproducible and automated ML pipeline

-> Track experiments and models using a model registry

-> Expose recommendations through a production-ready API

-> Monitor data and prediction drift

-> Enable automated retraining based on monitoring signals

🎯 Why this model is built (real-world motivation)

In real companies, recommendation systems are not built as a single model.

They are built to solve three practical problems:

1. Scalability

When the number of items is large, it is not possible to score all items for every user.

Therefore, real systems first generate a small candidate set.

This project implements:

-> a candidate generation model to retrieve a small subset of relevant items per user.

2. Personalization quality

A single collaborative filtering model is not enough to capture complex user behaviour.

In practice, a separate ranking model is used to learn:

-> user behaviour patterns

-> item popularity signals

-> interaction statistics

-> model prediction scores

This project uses a ranking model on top of generated candidates to improve final recommendation quality.

3. Production reliability

In real production systems:

-> models change

-> data changes

-> user behaviour changes

Therefore, this project focuses on:

-> experiment tracking

-> versioned models

-> monitoring

-> retraining pipelines

rather than only offline accuracy.

🧠 High-Level Architecture

User interaction data
        ↓
Data ingestion & validation
        ↓
Candidate generation model
        ↓
Candidate set per user
        ↓
Feature generation for ranking
        ↓
Ranking model
        ↓
Top-K recommendations
        ↓
FastAPI serving layer
        ↓
Prediction & input logging
        ↓
Monitoring & drift detection
        ↓
Automated retraining pipeline


🧩 System Design (Two-Stage Recommendation)
Stage 1 – Candidate Generation

Purpose:

-> Retrieve a small set of potentially relevant items for a user.

Model:

-> Collaborative filtering using matrix factorization.

Output:

-> Top-N candidate items per user.

Stage 2 – Ranking

Purpose:

-> Rank the candidate items using richer behavioural and statistical features.

Model:

-> Supervised learning model (e.g. Gradient Boosting / Logistic Regression).

Input features include:

-> user activity statistics

-> item popularity statistics

-> collaborative filtering prediction score

-> recency signals

Output:

-> Final ranked Top-K recommendations.

🔁 MLOps Lifecycle

RecoFlow implements a complete ML lifecycle:

-> Data ingestion and preprocessing

-> Training pipelines for:

-> candidate generation model

-> ranking model

-> Experiment tracking with MLflow

-> Model registry and versioning

-> Automated model promotion

-> API-based inference service

-> Monitoring with drift and data quality reports

-> Retraining triggered by monitoring signals

🛠️ Technology Stack

-> Python

-> pandas, numpy

-> scikit-surprise (candidate model)

-> scikit-learn / LightGBM (ranking model)

-> MLflow (experiment tracking & registry)

-> FastAPI (serving layer)

-> Evidently (monitoring & drift detection)

-> Docker

-> GitHub Actions (CI – optional)

📂 Project Structure

recoflow-mlops/
│
├── api/                          # Inference service
│   └── main.py
│
├── pipelines/                    # Orchestration entry points
│   ├── train_candidate_pipeline.py
│   ├── train_ranker_pipeline.py
│   └── monitoring_pipeline.py
│
├── src/
│   │
│   ├── config/                   # Centralised configs
│   │   └── settings.py
│   │
│   ├── ingestion/                # Data loading
│   │   └── load_interactions.py
|   |   └── ingest.py
│   │
│   ├── validation/               # Data quality & schema checks
│   │   └── interaction_checks.py
│   │
│   ├── splitting/                # Train/val/test splitting logic
│   │   └── user_split.py
│   │
│   ├── candidate_generation/     # Stage-1 models
│   │   ├── train_cf.py
│   │   └── generate_candidates.py
│   │
│   ├── ranking/                  # Stage-2 models
│   │   ├── feature_builder.py
│   │   ├── train_ranker.py
│   │   └── evaluate_ranker.py
│   │
│   ├── serving/                  # Core inference logic (not HTTP)
│   │   └── recommender.py
│   │
│   ├── monitoring/               # Drift, data quality, reports
│   │   └── drift.py
│   │
│   └── common/                   # Shared utilities
│       ├── logging.py
│       └── io.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── monitoring/
│
├── models/                       # Local model artifacts (optional)
│
├── notebooks/                    # Exploration only
│
├── docker/
│   ├── api.Dockerfile
│   └── training.Dockerfile
│
├── scripts/                      # One-off utility scripts
│
├── tests/
│
├── logs
|     └── recoflow.log
├── .gitignore
├── requirements.txt
└── README.md


📊 Evaluation Strategy

The project uses two types of evaluation:

Candidate model

-> RMSE (for collaborative filtering quality)

Final recommendation quality

-> Precision@K

-> Recall@K

-> NDCG@K

This reflects real-world ranking-based evaluation instead of only regression metrics.

🌐 API Endpoints

The serving layer exposes:

-> POST /recommend

-> POST /recommend/batch

-> POST /feedback

-> GET /model/info

The API loads the currently promoted models from the model registry.

📈 Monitoring

The system monitors:

-> input feature distributions

-> user activity distributions

-> prediction distributions

-> data drift between training and production data

Drift and quality reports are generated automatically.

🔄 Automated Retraining

When drift or performance degradation is detected:

-> training pipelines are triggered

-> new models are trained and evaluated

-> the best model is registered and promoted automatically

🧪 Dataset

The initial implementation uses the MovieLens dataset as interaction data.

The dataset is treated as a generic user–item interaction dataset to simulate real-world recommendation scenarios.

🧑‍💻 Author

Ridhul Krishna P
Capstone project focused on real-world recommendation systems and MLOps engineering.

⭐ Summary

RecoFlow is not a demo recommender.

It is a complete, production-style recommendation platform designed to demonstrate:

-> real system architecture

-> ML engineering practices

-> MLOps workflows

-> deployment and monitoring strategies

in a single end-to-end project.