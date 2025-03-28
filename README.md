# ACNH Clothing Price Prediction

This project is a machine learning pipeline designed to predict the sell price of clothing items in the game **Animal Crossing: New Horizons (ACNH)**. The pipeline includes data wrangling, cleaning, preprocessing, model training, and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
- [Workflow](#workflow)
  - [1. Data Wrangling](#1-data-wrangling)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Data Cleaning](#3-data-cleaning)
  - [4. Preprocessing](#4-preprocessing)
  - [5. Model Training](#5-model-training)
  - [6. Model Evaluation](#6-model-evaluation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to predict the **sell price** of clothing items in ACNH based on various features such as category, style, color, and more. The pipeline uses PyTorch for building and training a neural network model, with preprocessing and analysis performed using Python libraries like Pandas, Scikit-learn, and Seaborn.

---

## Directory Structure

```
├── 01_data_wrangling.ipynb       # Combines raw datasets into a single dataframe
├── 02_eda.ipynb                  # Exploratory Data Analysis
├── 03_data_cleaning.ipynb        # Handles missing values, outliers, and text normalization
├── 04_preprocessing.ipynb        # Encodes categorical variables, scales data, and splits train/test sets
├── 05_model_train.ipynb          # Defines and trains the PyTorch model
├── 06_model_evaluation.ipynb     # Evaluates the trained model
├── data/
│   ├── combined_data.csv         # Combined dataset
│   ├── clean/
│   │   ├── no_outliers.csv       # Cleaned dataset without outliers
│   │   ├── w_outliers.csv        # Cleaned dataset with outliers
│   ├── preprocessed/
│       ├── embedding_sizes.pkl   # Embedding sizes for categorical features
│       ├── preprocessing.pkl     # Scalers and encoders
│       ├── train_test_data.pkl   # Train/test split data
├── models/
│   ├── acnh_model.py             # PyTorch model definition
│   ├── model_versions_log.csv    # Log of trained model versions
│   ├── configs/                  # Model configuration files
│   ├── versions/                 # Saved model weights
```

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rainalexotl/acnh_clothing_price_prediction.git
   cd acnh_clothing_price_prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the raw datasets are placed in the `data/` directory. They can be found at [Kaggle](https://www.kaggle.com/datasets/jessicali9530/animal-crossing-new-horizons-nookplaza-dataset).

---

## Workflow

### 1. Data Wrangling
- Combines raw datasets (e.g., `accessories.csv`, `bags.csv`) into a single dataframe.
- Adds a `Category` column to identify the type of clothing.
- Saves the combined data to `data/combined_data.csv`.

### 2. Exploratory Data Analysis (EDA)
- Analyzes the distribution of the target variable (`Sell`) and other features.
- Identifies columns to drop and handles outliers.

### 3. Data Cleaning
- Drops unnecessary columns (e.g., `Name`, `Variation`, `Buy`).
- Fills missing values (e.g., `Mannequin Piece` is filled with "No").
- Saves cleaned datasets:
  - `data/clean/w_outliers.csv` (with outliers).
  - `data/clean/no_outliers.csv` (without outliers).

### 4. Preprocessing
- Encodes categorical variables using `.cat.codes`.
- Scales the target variable (`Sell`) using `MinMaxScaler`.
- Splits the data into training and testing sets.
- Saves preprocessed data and metadata.

### 5. Model Training
- Defines the model architecture using `ACNHModel` in [acnh_model.py](models/acnh_model.py).
- Trains the model using Mean Squared Error (MSE) loss and Adam optimizer.
- Logs training and validation losses over epochs.
- Saves the trained model and configuration.

### 6. Model Evaluation
- Evaluates the model's performance using metrics like RMSE.
- Visualizes predictions vs. actual values and residuals.
- Analyzes embeddings to understand feature relationships.

---

## Model Architecture

The model is a feedforward neural network implemented in PyTorch. It includes:
- **Embedding layers** for categorical features.
- **Fully connected layers** with ReLU activation.
- **Dropout** and **Batch Normalization** for regularization.

See [acnh_model.py](models/acnh_model.py) for the full implementation.

---

## Results

- **MSE**: 0.0230 on the test set.
- **RMSE**: 0.1517 on the test set. (**Mean**=0.3922, **Std**=0.1935)
- The model performs well on most categories but struggles with high-priced outliers.
- Embedding analysis shows meaningful relationships between categories.

---

## Acknowledgments

- Dataset: [Kaggle - Animal Crossing New Horizons Catalog](https://www.kaggle.com/datasets/jessicali9530/animal-crossing-new-horizons-nookplaza-dataset)
- Libraries: PyTorch, Pandas, Scikit-learn, Seaborn, Matplotlib