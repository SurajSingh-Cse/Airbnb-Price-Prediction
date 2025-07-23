# 🏠 Airbnb Price Prediction - NYC

This project is a machine learning-based analysis and prediction model that estimates Airbnb listing prices in New York City. Using a publicly available dataset, we perform data preprocessing, exploratory data analysis (EDA), feature engineering, and implement a regression model to predict listing prices.

---

## 📌 Project Objectives

* Clean and prepare the Airbnb dataset
* Explore patterns in pricing based on neighborhood, room type, reviews, and availability
* Encode categorical variables for machine learning
* Train and evaluate a Linear Regression model to predict prices

---

## 📂 Dataset

**Source:** [Kaggle - NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

**Features include:**

* Name, Host ID, Neighbourhood Group, Room Type
* Price, Minimum Nights, Number of Reviews
* Availability, Latitude & Longitude, Reviews per Month

---

## 🛠️ Technologies Used

* Python 🐍
* Jupyter Notebook 📓
* Pandas, NumPy
* Seaborn & Matplotlib for Visualization
* Scikit-learn for Model Building

---

## 🔍 Exploratory Data Analysis (EDA)

* Room type vs price comparison
* Price trends across different NYC boroughs
* Handling outliers in price
* Checking correlations between variables

---

## 🧹 Data Cleaning

* Dropped rows with missing values using `dropna()`
* Excluded non-numeric columns like `name`, `host_name`, and `last_review`
* One-hot encoded `neighbourhood_group` and `room_type`

---

## 📈 Model Building

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

✅ Model evaluation is done using **RMSE** and **R² Score**.

---

## 📊 Results

* RMSE: (Example: 108.34)
* R² Score: (Example: 0.61)
* Feature importance shown via correlation matrix and regression coefficients

---

## 📌 Conclusion

* Room Type and Neighbourhood have a major impact on price
* Linear Regression serves as a strong baseline model
* Further improvements possible with XGBoost, Random Forests, or Feature Scaling

---

## 🧠 Author

**Suraj Singh**



📫 [LinkedIn](https://www.linkedin.com/in/surajsingh-cse) 


🌐 [GitHub]([(https://github.com/SurajSingh-Cse/Airbnb-Price-Prediction))

---

## 🔖 Tags

`#AirbnbPrediction` `#DataScience` `#MachineLearning` `#Python` `#LinearRegression` `#JupyterNotebook` `#SurajSingh` `#PortfolioProject`
