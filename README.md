
# Titanic Project  
  
This project implements a complete machine-learning pipeline for predicting passenger survival on the Titanic dataset.  
It includes automated preprocessing, feature engineering, cross-validated model training, and a unified inference system.  
  
***Features***  
  
- Data preprocessing pipeline  
- Cross-validation framework  
- Multiple models supported:  
    - Logistic Regression  
    - Ridge Classifier  
    - KNN  
    - Decision Tree  
    - Random Forest  
    - CatBoost  
    - LightGBM  
    - XGBoost  
    - Custom Deep Neural Network (DNN) model  
  
- Checkpointing  
- Inference module  
  
  
***Project Structure***  
```  
project/  
│  
├── solver.py        # Model training and inference manager  
├── dataset.py       # Preprocessing, feature engineering, CV folds  
├── dnn.py           # Custom deep neural network  
├── config.py        # Configuration for models and paths  
├── main.py          # Train / predict entry point  
└── checkpoints/     # Saved model weights and pipelines  
```  
  
***TODO***  
- Add visualizations  
- Add stacking, voting  
- Add automated grid search (currently you should tune params yourself)  
  
***My Results***  

| Model          | Model ID | Accuracy   |  
|----------------|----------|------------|  
| ridge          | 0        | 0.966507   |  
| log_reg        | 0        | 0.944976   |  
| random_forest  | 0        | 0.854067   |  
| lgbm           | 0        | 0.799043   |  
| DNN            | 0        | 0.782297   |