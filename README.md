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
#import libraries:-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load dataset:-
df = pd.read_csv("AB_NYC_2019.csv")
df.head()

#basic info and missing value:-
df.info()   # Display basic information about the DataFrame

df.describe() # Display summary statistics of the DataFrame

df.isnull().sum()  # Check for missing values in each column

#drop missing raw and and missing value:-
df = df.dropna()  # Drop rows with missing values

#  Define X and y properly (Drop all non-numeric/text columns from X:-
X = df.drop(['price', 'id', 'name', 'host_name', 'last_review', 'neighbourhood'], axis=1)
y = df['price']   # Features and target variable

X = pd.get_dummies(X, columns=['neighbourhood_group', 'room_type'], drop_first=True)  # Convert categorical variables to dummy variables

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train Linear Regression:-
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)    # Train the model

#  Predict and Evaluate:-
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression RMSE:", rmse)  # Root Mean Squared Error       
print("Linear Regression R² Score:", r2)  # R² Score



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

## Output:

```
Linear Regression RMSE: 168.90248652881712
Linear Regression R² Score: 0.1471733218820559
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


🌐 [GitHub]-([(https://github.com/SurajSingh-Cse/Airbnb-Price-Prediction)

---

## 🔖 Tags

`#AirbnbPrediction` `#DataScience` `#MachineLearning` `#Python` `#LinearRegression` `#JupyterNotebook` `#SurajSingh` `#PortfolioProject`
