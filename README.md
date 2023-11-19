# rent-prices

## Introduction
For this project, I have created a tool to get an estimated rent price for certain neighborhoods of Madrid, based on the apartment parameters selected by the user. The web app has been built using Streamlit and I have used a linear regression model in order to predict the rent price.


## Steps

### 1. Data collection
I have used Idealista API to get a list of apartments, including their characteristics and rent price. The data has being downloaded into a .csv file.

### 2. EDA
Understanding of the dataset: columns, types of data.

### 3. Data cleaning and formatting
Deleting unnecessary info and transforming data to create the regression model.

### 4. Feature selection
Choosing the variables to be used in the model and encoding the categorical ones.

### 5. Linear Regression Model
The model has 7 variables (size, exterior, hasLift, hasParkingSpace, isFloorZero, propertyType, neighborhood) and has been trained using 30% of the data. An R-squared of 0.72 was obtained.

### 6. Web app development
Using Streamlit, I built an interface where the user can use the different parameters (the variables used in the model) to get an estimated rent price based on these inputs.