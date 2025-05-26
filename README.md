# ACNH Clothing Price Prediction

Predicting in-game clothing prices in **Animal Crossing: New Horizons (ACNH)** using PyTorch & modern EDA techniques.

---

## Project Overview

This project demonstrates an end-to-end machine learning pipeline on tabular data using PyTorch. I designed, implemented, and evaluated a model to predict clothing “sell” prices in Animal Crossing: New Horizons, using real game catalog data. The project covers advanced EDA, feature engineering, neural network modeling with embeddings, and insightful result analysis.

**Why does this matter?**  
Predicting item values helps understand game economies and supports building tools for players to optimize trading and collecting strategies. 
*Also, because someone has to stand up to Tom Nook’s questionable pricing, and it might as well be a neural network.*

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
   git clone https://github.com/rainalexotl/ml-economics-animal-crossing.git
   cd acnh_clothing_price_prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the raw datasets are placed in the `data/` directory. They can be found at [Kaggle](https://www.kaggle.com/datasets/jessicali9530/animal-crossing-new-horizons-nookplaza-dataset).

---

## Workflow

1. **Data Wrangling:** Merge multiple clothing datasets, add category labels.
2. **EDA:** Analyze distributions, identify outliers, and explore relationships.
3. **Data Cleaning:** Drop unnecessary columns, fill missing values, create clean data splits.
4. **Preprocessing:** Encode categorical variables, scale features, split train/test sets.
5. **Model Training:** 
    - PyTorch NN with embedding layers for categorical data.
    - Training with MSE loss and Adam optimizer.
6. **Evaluation:** 
    - Metrics: MSE, RMSE.
    - Visualizations: Predictions vs actuals, residuals, embedding analysis.

---

## Model Architecture

- **Embeddings** for categorical features (e.g., category, style, color).
- **Feedforward layers** with ReLU, batch norm, and dropout.
- Full implementation: [`models/acnh_model.py`](models/acnh_model.py)

---

## Results

- **Test MSE**: 0.0230
- **Test RMSE**: 0.1517 (**Mean**=0.3922, **Std**=0.1935)
- Model performs well overall, but is less accurate for high-priced outliers.
- Embedding analysis reveals meaningful category relationships.

---

## Insights & Next Steps

- **Strengths:** Demonstrates deep EDA, categorical embeddings, and custom PyTorch modeling.
- **Limitations:** Struggles with rare, high-priced items due to data imbalance.
- **Future Work:** 
    - Try more advanced architectures (e.g., TabNet, GBDT+NN hybrid).
    - Add item images for multimodal learning.
    - Deploy as a simple web app for in-game use.

---

## Acknowledgments

- Dataset: [Kaggle - Animal Crossing New Horizons Catalog](https://www.kaggle.com/datasets/jessicali9530/animal-crossing-new-horizons-nookplaza-dataset)
- Libraries: PyTorch, Pandas, Scikit-learn, Seaborn, Matplotlib
