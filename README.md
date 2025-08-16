# 🏠 Ultra-Boosted Stacked Ensemble for House Prices

This repository contains my solution for the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.  

The approach is a **stacked ensemble** that combines:
- Neural Networks (TensorFlow/Keras)
- XGBoost
- LightGBM
- CatBoost  

with **10-fold cross-validation**, **feature pruning**, **stacking with a Ridge meta-model**, and **optimized convex blending**.

📊 **Current Kaggle Leaderboard Rank**: **273rd out of 4306 competitors** 🎉  

---

## 🚀 Pipeline Overview

### 1. Data Preprocessing
- Load train/test datasets.
- Feature engineering:
  - `TotalSF` (Total Square Footage)  
  - `TotalBath` (Full/Half Baths including Basement)  
  - `HouseAge`, `RemodAge`  
  - Interaction terms (`GrLivArea * OverallQual`, etc.)  
- Missing value handling:
  - Numeric → median imputation  
  - Categorical → `"Missing"`

### 2. Feature Transformation
- `StandardScaler` for numerical features  
- `OneHotEncoder` for categorical features  
- Create a unified processed dataset (`X_all`, `X_test`)

### 3. Feature Importance Pruning
- Fit a LightGBM probe model  
- Keep the **top-K features** explaining ≥95% cumulative importance (minimum of 300 features retained)

### 4. Base Models
- **Neural Network (Keras/TensorFlow)**  
  - 3 hidden layers (512 → 256 → 128)  
  - BatchNorm + Dropout  
  - Adam optimizer + Huber loss  
- **XGBoost** (5000 trees, depth=4)  
- **LightGBM** (5000 trees, 32 leaves)  
- **CatBoost** (5000 iterations, depth=6)

### 5. 10-Fold Cross-Validation
- Out-of-Fold (OOF) predictions generated for all 4 models  
- Neural Net bagging (2 seeds per fold for stability)  
- Store both OOF predictions and test predictions per fold

### 6. Stacking
- Train **Ridge regression** on OOF predictions  
- Use it as a **meta-model** for stacking

### 7. Optimized Blending
- Perform a grid search over convex combinations of model weights  
- Ensure non-negative weights summing to 1  
- Pick weights that minimize OOF RMSE

### 8. Model Selection
- Compare Ridge stacking vs optimized blending  
- Choose the approach with lower OOF RMSE

### 9. Final Submission
- Back-transform predictions from log-space (`expm1`)  
- Save predictions as `submission.csv`

---

## 📈 Performance
- Out-of-Fold RMSE diagnostics are printed for each base model and the stacked/blended models  
- Final predictions achieve a **leaderboard rank of 273 / 4306** (AT the current moment, competition is always running so leaderboard may change later)

---

## ⚙️ Requirements
- Python 3.8+
- Libraries:
  - `numpy`, `pandas`, `scikit-learn`
  - `tensorflow`
  - `xgboost`
  - `lightgbm`
  - `catboost`


