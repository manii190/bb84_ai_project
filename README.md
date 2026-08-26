# AI-Based Eavesdropping Detection in BB84 Quantum Key Distribution

A machine-learning research project exploring the detection of eavesdropping attacks in **BB84 Quantum Key Distribution (QKD)** using simulated quantum communication data.

This project was developed as part of a **Research Internship at the University of Hyderabad, School of Physics**.

## Overview

Quantum Key Distribution provides a method for securely exchanging cryptographic keys using principles of quantum mechanics. The **BB84 protocol** can reveal the presence of an eavesdropper because interception introduces measurable disturbances into the quantum channel.

This project investigates whether **machine-learning models can identify eavesdropping behavior from characteristics of simulated BB84 communication sessions**.

The workflow combines:

**BB84 Simulation → Feature Engineering → Dataset Generation → Machine Learning → Attack Detection → Model Evaluation**

## Objectives

* Simulate normal and attacked BB84 communication sessions
* Model intercept-resend eavesdropping attacks
* Introduce realistic channel effects such as noise and photon loss
* Extract useful security-related features from each session
* Train machine-learning classifiers to distinguish normal communication from attacks
* Compare models using multiple performance metrics
* Analyze which features contribute most to attack detection

## BB84 Simulation

The Python-based simulation models major stages of the BB84 protocol, including:

* Random bit generation
* Random basis selection
* Quantum-state preparation
* Photon transmission
* Channel noise
* Photon loss
* Measurement
* Basis reconciliation
* Key sifting
* Intercept-resend eavesdropping

Both **normal** and **attack** scenarios are generated for machine-learning analysis.

## Features

The machine-learning pipeline uses security and channel characteristics derived from simulated BB84 sessions, including:

* Quantum Bit Error Rate (QBER)
* Noise level
* Photon loss
* Basis-specific QBER measurements
* QBER variation
* Additional channel/detection statistics

These features allow the models to learn patterns associated with eavesdropping.

## Machine Learning Models

Multiple classification algorithms are evaluated, including:

* Random Forest
* Decision Tree
* Logistic Regression
* Support Vector Machine (SVM)
* XGBoost

The project compares different models rather than relying on a single classifier.

## Model Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* False Positive Rate
* False Negative Rate
* Training Time
* Feature Importance

Learning-curve experiments are also used to study how model performance changes as the amount of training data increases.

## Technology Stack

### Programming

* Python

### Data & Machine Learning

* NumPy
* Pandas
* Scikit-learn
* XGBoost

### Visualization

* Matplotlib

### Research Areas

* Machine Learning
* Quantum Computing
* Quantum Cryptography
* Quantum Key Distribution
* BB84
* Cybersecurity

## Project Pipeline

```text
BB84 Protocol Simulation
          ↓
Normal / Attack Sessions
          ↓
Channel Noise & Photon Loss
          ↓
Feature Extraction
          ↓
Dataset Generation
          ↓
Machine Learning Training
          ↓
Attack Classification
          ↓
Performance Evaluation
```

## Research Context

Traditional BB84 security analysis often relies heavily on metrics such as QBER to identify abnormal behavior.

This project explores an additional approach: using multiple characteristics of a simulated quantum communication session as inputs to machine-learning classifiers.

The goal is to investigate whether AI-assisted analysis can help distinguish between normal channel disturbances and behavior associated with simulated eavesdropping attacks.

## Repository Structure

```text
bb84_ai_project/
│
├── src/                 # BB84 simulation and ML code
├── models/              # Model-related files
├── results/             # Experimental results
├── plots/               # Generated visualizations
├── requirements.txt     # Python dependencies
└── README.md
```

*The exact repository structure may vary as the project continues to be organized.*

## Running the Project

Clone the repository:

```bash
git clone https://github.com/manii190/bb84_ai_project.git
cd bb84_ai_project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install the required packages:

```bash
pip install -r requirements.txt
```

Run the relevant simulation or training scripts from the repository.

## Data

Large generated datasets are not intended to be stored directly in this GitHub repository.

The datasets are produced through the BB84 simulation pipeline and can be regenerated for experimentation.

This keeps the repository lightweight while preserving the code required to reproduce the research workflow.

## Future Work

Potential extensions include:

* Testing additional eavesdropping strategies
* Investigating more realistic quantum-channel models
* Evaluating additional machine-learning architectures
* Improving feature engineering
* Studying model generalization across different noise environments
* Exploring real or hardware-generated QKD data
* Comparing ML-based detection with traditional statistical approaches

## Disclaimer

This repository represents a **research and simulation project**. Results are based on simulated BB84 communication environments and should not be interpreted as validation for production quantum communication systems.

## Author

**Mani Abhiram Reddy Venna**

Artificial Intelligence Undergraduate
University of Arizona

Research Internship — University of Hyderabad, School of Physics

## Connect

* GitHub: `manii190`
* LinkedIn: Add your LinkedIn profile URL here
