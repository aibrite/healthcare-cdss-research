# Bayesian Network for Healthcare CDSS

R&D Work For Bayesian Networks under the topic Probabilistic Graphical Models for Clinical Decision Support Systems.

## Overview

This project implements Bayesian Networks for medical diagnosis, focusing on liver disease and heart disease prediction. It provides both standalone diagnostic models and a web-based API for real-time inference, along with interactive visualizations.

## Features

- **Liver Disease Diagnosis**: Probabilistic model based on alcohol abuse history, AST/ALT ratio, and GGT levels
- **Heart Disease Diagnosis**: Model using chest pain type, exercise-induced angina, and resting ECG results
- **REST API**: Flask-based backend for real-time diagnostic queries
- **Interactive Visualizations**: Web-based network visualizations with conditional probability tables
- **Flexible Architecture**: Support for both expert-defined and data-driven models

## Project Structure

```
bayesian-network/
├── main.py                    # Core liver disease model with predefined CPDs
├── heart-disease.py           # Heart disease model using ML estimation
├── backend_server.py          # Flask REST API server
├── visualize.py              # Interactive HTML visualization generator
├── *.html                    # Generated network visualizations
└── pyproject.toml            # Project dependencies
```

## Installation

### Prerequisites
- Python >= 3.10
- UV package manager (recommended) or pip

### Setup

1. **Clone and navigate to the project**:
   ```bash
   cd bayesian-network
   ```

2. **Install dependencies using UV**:
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install flask flask-cors pandas pgmpy
   ```

## Usage

### 1. Run Standalone Models

**Liver Disease Model:**
```bash
uv run main.py
```

**Heart Disease Model:**
```bash
uv run heart-disease.py
```

### 2. Start the API Server

```bash
uv run backend_server.py
```

The server will start on `http://localhost:5000` with the following endpoints:

#### Available API Endpoints

- `GET /` - Health check
- `POST /query` - Perform probabilistic inference

**Example API Usage:**
```bash
# Query liver disease probability
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "variables": ["Diagnosis"],
    "evidence": {
      "AST_ALT_Ratio": "> 2.0",
      "GGT": "Elevated"
    }
  }'
```

### 3. Generate Visualizations

```bash
uv run python visualize.py
```

This generates interactive HTML files showing:
- Network structure with nodes and edges
- Conditional Probability Tables (CPTs)
- Interactive probability calculations

## Models

### Liver Disease Model

**Variables:**
- `Alcohol_Abuse`: {No, Yes}
- `Liver_Disease`: {No, Yes}  
- `AST_ALT_Ratio`: {< 1.0, 1.0-2.0, > 2.0}
- `GGT`: {Normal, Elevated}
- `Diagnosis`: {No Disease, Disease}

### Heart Disease Model

**Variables:**
- `ChestPain`: {typical, atypical, non-anginal, asymptomatic}
- `ExerciseAngina`: {no, yes}
- `RestECG`: {normal, abnormal}
- `HeartDisease`: {absent, present}

**Network Structure:**
```
ChestPain     → HeartDisease
ExerciseAngina → HeartDisease  
RestECG       → HeartDisease
```

## API Reference

### POST /query

Perform probabilistic inference on the Bayesian network.

**Request Body:**
```json
{
  "variables": ["variable_to_query"],
  "evidence": {
    "observed_variable": "observed_state"
  }
}
```

**Response:**
```json
{
  "result": {
    "variable_name": {
      "state1": probability1,
      "state2": probability2
    }
  }
}
```

## Example Queries

### Liver Disease Scenarios

1. **High-risk patient:**
   ```python
   # High AST/ALT ratio + Elevated GGT
   query = inference.query(
       variables=["Diagnosis"], 
       evidence={"AST_ALT_Ratio": "> 2.0", "GGT": "Elevated"}
   )
   ```

2. **Low-risk patient:**
   ```python
   # No alcohol abuse + Normal GGT
   query = inference.query(
       variables=["Diagnosis"], 
       evidence={"Alcohol_Abuse": "No", "GGT": "Normal"}
   )
   ```

### Heart Disease Scenarios

1. **Typical symptoms:**
   ```python
   query = inference.query(
       variables=["HeartDisease"],
       evidence={"ChestPain": 0, "ExerciseAngina": 1, "RestECG": 1}
   )
   ```

## Research Applications

This implementation supports research in:

- **Clinical Decision Support Systems (CDSS)**
- **Probabilistic Medical Diagnosis**
- **Uncertainty Quantification in Healthcare**
- **Evidence-Based Medicine**
- **Risk Assessment and Stratification**

## Technical Details

### Dependencies

- **pgmpy**: Probabilistic graphical models library
- **Flask**: Web framework for REST API
- **pandas**: Data manipulation and analysis
- **Flask-CORS**: Cross-origin resource sharing

### Model Validation

All models include validation checks to ensure:
- Network structure is acyclic
- Conditional probability distributions sum to 1
- Evidence consistency
- Proper variable cardinalities
