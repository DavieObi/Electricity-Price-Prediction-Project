# Electricity Price Prediction Project

This project details the process of building and evaluating a machine learning model to predict electricity prices (`SMPEP2`) using a dataset that includes environmental, market, and time-series data. The primary goal was to develop a robust model capable of accurately forecasting electricity prices.

## Dataset

The dataset used in this project was sourced from a public GitHub repository. It contains 38,014 entries with 18 columns, including:
* **Time-series data:** `DateTime`, `PeriodOfDay`, `DayOfWeek`, `Month`, `Year`, etc.
* **Environmental data:** `ORKTemperature`, `ORKWindspeed`, `CO2Intensity`.
* **Market data:** `ForecastWindProduction`, `ActualWindProduction`, `SystemLoadEA`, `SMPEA`, `SystemLoadEP2`, `SMPEP2`.

## Methodology

The project followed a structured machine learning workflow, as outlined below:

### 1. Data Cleaning and Preprocessing
The initial dataset required several cleaning steps to be suitable for modeling:
* **Missing Value Handling:** The `Holiday` column, which had a very high proportion of missing values, was dropped. Rows with any remaining missing values were also dropped using `dropna()`.
* **Data Type Conversion:** Numerous columns containing numerical data were stored as `object` (string) types. These were converted to `float64` to enable mathematical operations. The `DateTime` column was converted to a proper `datetime` object for time-series analysis.

### 2. Feature Engineering
New features were created to help the model better understand patterns in the data:
* **Time-Based Features:** New columns for `Hour` were extracted from the `DateTime` column.
* **Lag and Rolling Features:** A lag feature (`Lag_SMPEA`) and a rolling mean feature (`Rolling_SMPEA_3h`) were created from the `SMPEA` column to capture temporal dependencies.
* **Interaction Feature:** An interaction feature (`Temp_Wind_Interaction`) was created by multiplying `ORKTemperature` and `ORKWindspeed`.

### 3. Model Building & Evaluation
A `RandomForestRegressor` was chosen as the primary model due to its strong performance on complex, non-linear datasets.

* **Baseline Model:** A simple `LinearRegression` model was initially trained, which yielded an R-squared of approximately 0.276, indicating the need for a more complex model.
* **Primary Model:** The `RandomForestRegressor` was trained on the full feature set. This model achieved a significantly better performance, with an R-squared of approximately 0.552.
* **Feature Importance Analysis:** A feature importance plot was generated to identify the key predictors for the `SMPEP2` price. The most important features were found to be `SMPEA`, `SystemLoadEP2`, and the engineered time-series features.

### 4. Cross-Validation for Robustness
To validate the model's performance and ensure the results were not due to a random split of the data, a K-Fold cross-validation was performed. Two models were compared:
* **Model A (Full Features):** Used all available features.
* **Model B (Lean Features):** Used a reduced feature set, dropping some of the highly correlated variables.

The cross-validation results confirmed that **Model A (Full Features)** was the most robust and accurate model, achieving:
* **Average R-squared:** 0.6122
* **Average MAE:** 9.1770

## Conclusion

The `RandomForestRegressor` model, trained on the full feature set including the engineered time-series and interaction features, proved to be a highly effective predictor of electricity prices. The cross-validation confirmed that this model is both accurate and consistent. Further improvements could be explored through hyperparameter tuning or by using more advanced time-series modeling techniques.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Seaborn
