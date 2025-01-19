# Bengaluru House Price Prediction

This project uses machine learning techniques to predict house prices in Bangalore based on factors such as location, size, BHK, and bath. The models implemented in this project include Ridge, XGBoost, and other regression algorithms to provide the best predictions for house prices.

## Authors

- [@SamiUllah568](https://github.com/SamiUllah568)

## Table of Contents

* [Project Overview](#project-overview)
* [Technologies Used](#technologies-used)
* [Getting Started](#getting-started)
* [Model Training](#model-training)
* [Model Evaluation](#Model-Evaluation)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Usage](#usage)
* [Model Saving and Loading](#model-saving-and-loading)
* [Results](#results)
* [Conclusion](#conclusion)
* [License](#license)

## Project Overview

The goal of this project is to build a predictive model that can estimate the price of a house based on its location, size, BHK, and number of bathrooms. The dataset consists of various features that are processed, cleaned, and used to train regression models.

### Key Features of the Dataset:
- **Location**: The location of the house
- **Total_Sqft**: The total square footage of the house
- **Bath**: The number of bathrooms
- **BHK**: The number of bedrooms
- **Price**: The price of the house (target variable)

## Technologies Used
- **Python**
- **Scikit-learn** (for machine learning models)
- **XGBoost**
- **Matplotlib** (for data visualization)
- **Seaborn** (for data visualization)
- **Pandas** (for data manipulation)
- **NumPy** (for numerical computations)
- **Pickle** (for saving and loading models)

## Getting Started

To get started with this project:

1. Clone the repository:

   ```bash
   git clone https://github.com/SamiUllah568/Bengaluru-House-Price-Prediction.git



## Model Training
Various machine learning regression models have been used to predict the house prices, including Ridge Regression and XGBoost, and model performance was evaluated using metrics like R-squared and Mean Squared Error.

the following regression models were evaluated:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Support Vector Machine Regression
- Decision Tree Regressor
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- CatBoost Regressor


## Hyperparameter Tuning

Hyperparameter tuning was performed on Ridge and XGBoost models to improve performance. A grid search was used to find the optimal parameters for the models, and cross-validation was employed to prevent overfitting.

For example, for Ridge regression, the following parameters were tuned:
- alpha
- fit_intercept
- max_iter
- solver

For XGBoost, parameters like the following were optimized:
- learning_rate
- max_depth
- n_estimators
- subsample


## Model Evaluation

### Train Data:
- **Mean Squared Error (MSE)**: 499.78
- **R-squared Score**: 0.93

### Test Data:
- **Mean Squared Error (MSE)**: 825.79
- **R-squared Score**: 0.84

### Observations:
- The model performed well on the training data with a high R-squared score (0.93), indicating that it explains a large proportion of the variance in the data.
- The test data also shows a good R-squared score (0.84), but there is a noticeable drop from the training data, which may indicate some overfitting.
- The Mean Squared Error (MSE) on the test data is higher than that on the training data, confirming that the model has a slight decrease in performance when applied to unseen data.

This evaluation suggests that the model is robust but could potentially be fine-tuned further for better generalization.


## Usage
Once the model is trained, you can predict house prices by providing the input features such as location, square footage, BHK, and bath.

### Example:
```python
def predict_price(location, total_sqft, bath, BHK):
    loc_index = np.where(x_train.columns == location)[0][0]

    x_input = np.zeros(len(x_train.columns))
    x_input[0] = total_sqft
    x_input[1] = bath
    x_input[2] = BHK

    if loc_index >= 0:
        x_input[loc_index] = 1

    return model_XGB.predict([x_input])[0]
```

## Model Saving and Loading

After training the model, it was saved using the pickle module for later use.

### Saving the model:

```python
import pickle
pickle.dump(model_XGB, open("Banglore_House_price_Prediction_model.pkl", "wb"))

Loading the model:

Banglore_House_price_Prediction_model = pickle.load(open("Banglore_House_price_Prediction_model.pkl", "rb"))

```
## Results

### Model Evaluation:
XGBoost performed slightly better than Ridge Regression, with better R-squared and lower Mean Squared Error on both training and test datasets. Ridge Regression also performed well, but XGBoost was chosen as the final model for house price prediction.

### Metrics:
Mean Squared Error (MSE) and R-squared Score were used to evaluate model performance on the training and testing datasets.

## Conclusion
This project demonstrates how to apply machine learning techniques for house price prediction. The best performing model was XGBoost, which outperformed Ridge Regression in terms of accuracy and generalization.

## License
This project is open-source and available under the MIT License. See the LICENSE file for more information.
