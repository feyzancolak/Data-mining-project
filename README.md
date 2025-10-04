

# 🎵 Music Popularity Across Borders Using DM/ML Techniques

This project, developed for the **Data Mining and Machine Learning** course at the **University of Pisa**, investigates how musical tracks gain **popularity across different countries** using advanced **data mining** and **machine learning** techniques.

---

## 📝 Overview

The primary goal of this project is to analyze and model how music popularity spreads across borders.
By leveraging **data mining** and **machine learning** methods, the system identifies the main factors that influence international success and predicts whether a song popular in one country is likely to become popular in others.

The project combines **feature extraction**, **clustering**, **predictive modeling**, and **correlation analysis** to uncover insights about global listening trends and cultural similarities between countries.

---

## ✨ Key Features

* **Cross-Border Popularity Analysis**: Identify and visualize how songs perform across different countries and regions.
* **Feature Engineering**: Incorporates diverse musical and contextual attributes (e.g., tempo, energy, danceability, artist origin, language, genre).
* **Predictive Modeling**: Uses classification and regression algorithms to forecast international popularity.
* **Clustering & Segmentation**: Groups countries or tracks based on similarity in listening behaviors.
* **Correlation Analysis**: Evaluates relationships between song features and their cross-border performance.
* **Visualization Dashboards**: Displays global trends, heatmaps, and feature importance metrics for interpretation.
* **Data Preprocessing Pipeline**: Handles cleaning, normalization, encoding, and outlier detection.

---

## 🏗️ System Architecture

The project follows a **modular pipeline** architecture designed for scalability and interpretability.

### Core Components

1. **Data Collection**

   * Music dataset with features such as danceability, energy, loudness, tempo, language, artist origin, and country-level popularity metrics.
   * Data sources: Spotify Charts, Global Music Indexes, or open datasets.

2. **Data Preprocessing**

   * Missing value handling, normalization (Min-Max / Standard Scaler)
   * Encoding of categorical features (One-Hot, Label Encoding)
   * Aggregation and filtering per country or region

3. **Exploratory Data Analysis (EDA)**

   * Statistical summaries, correlation matrices, and trend visualizations
   * Detection of cross-country similarities using clustering (K-Means / Hierarchical)

4. **Machine Learning Models**

   * Supervised Learning: Random Forest, XGBoost, Logistic Regression, SVM
   * Unsupervised Learning: PCA, K-Means, DBSCAN for dimensionality reduction and grouping
   * Evaluation metrics: Accuracy, F1-Score, RMSE, R², and cross-validation

5. **Visualization & Insights**

   * Heatmaps of cross-country correlations
   * Scatter plots for feature impact
   * Model explainability via feature importance graphs

---

## 🛠️ Development Environment & Technologies

| Layer                    | Technologies          |
| ------------------------ | --------------------- |
| **Programming Language** | Python                |
| **Data Processing**      | pandas, numpy         |
| **Machine Learning**     | scikit-learn, xgboost |
| **Visualization**        | matplotlib, seaborn   |
| **Notebook Environment** | Jupyter               |
| **Version Control**      | Git, GitHub           |
| **IDE**                  | VS Code / JupyterLab  |

---

## ⚙️ System Configuration

### 1. Environment Setup

Clone the repository:

```bash
git clone https://github.com/feyzancolak/Data-mining-project.git
cd Data-mining-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the notebook or script:

```bash
jupyter notebook
# or
python main.py
```

### 2. Dataset Configuration

Place your dataset (e.g., `spotify_global_data.csv`) under the `/data` directory.
Make sure it includes:

* Track metadata (artist, name, genre)
* Popularity metrics (streams, rankings)
* Country identifiers

---

## 📊 Experimental Workflow

1. **Import & Preprocess Data**
   Clean, normalize, and encode musical features.
2. **Feature Selection**
   Identify key predictors of international success (e.g., danceability, tempo).
3. **Clustering Countries**
   Use K-Means or Hierarchical clustering to find regional music preference groups.
4. **Model Training**
   Predict cross-border success using classification/regression models.
5. **Evaluation**
   Analyze confusion matrices, precision/recall, or regression error metrics.
6. **Visualization**
   Generate dashboards and plots summarizing model performance and insights.

---

## 🔍 Results & Insights

* Discovered strong **correlations between cultural/geographical proximity** and musical taste.
* Identified **key audio features** (e.g., energy, tempo, speechiness) that influence global reach.
* Highlighted **distinct regional clusters** with similar listening preferences.
* Built predictive models capable of estimating **cross-border popularity** with high accuracy.
* Provided interpretable visualizations of the global music landscape.

---

## 🚀 Future Work

* Integration with **Spotify API** for real-time data retrieval
* Addition of **temporal trend analysis** (how songs gain/lose popularity over time)
* Use of **deep learning** (RNN/LSTM) for dynamic forecasting
* Sentiment analysis on lyrics to include linguistic influence
* Building a **web dashboard** for interactive exploration

---

## 👥 Contributors & Credits

* **Author:** Feyzan Çolak & Noemi Cherchi
  *M.S. in Artificial Intelligence & Data Engineering*
  University of Pisa

* **Course:** Data Mining and Machine Learning

* **Instructor:** Francesco Marcelloni

* **Dataset Sources:** Spotify, Kaggle Music Datasets


