import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Importing and cleaning data
houses = pd.read_csv('/Users/JAVIER/Ironhack/rent-prices/houses.csv')

houses.drop(['Unnamed: 0', 'thumbnail', 'numPhotos', 'showAddress',
       'url', 'distance', 'description', 'hasVideo', 'hasPlan', 'suggestedTexts', 'has3DTour', 'has360',
       'hasStaging', 'labels', 'superTopHighlight', 'topNewDevelopment',
       'externalReference', 'highlight', 'newDevelopmentFinished', 'province', 'municipality', 'country',
       'latitude', 'longitude'], axis=1, inplace=True)

houses.dropna(subset=['floor', 'exterior'], how='any', inplace=True)

# We need to extract the data stored in dictionaries from the column parkingSpace.
# To do so, we will create three new columns and set their values as False for the first two columns, and 0 for the third one.
# Using the function df.iterrows() we can get the values of each key from the column parkingSpace and add them to the new columns.

houses['isParkingSpaceIncludedInPrice'] = False
houses['hasParkingSpace'] = False
houses['parkingSpacePrice'] = 0.00

for index, rows in houses.iterrows():
    try:
        houses.loc[index, ['isParkingSpaceIncludedInPrice']] =  eval(rows['parkingSpace']).get('isParkingSpaceIncludedInPrice')
        houses.loc[index, ['hasParkingSpace']] =  eval(rows['parkingSpace']).get('hasParkingSpace')
        houses.loc[index, ['parkingSpacePrice']] =  eval(rows['parkingSpace']).get('parkingSpacePrice')
    except:
        pass

houses.parkingSpacePrice.fillna(0.0, inplace=True)

houses['totalPrice'] = houses['price'] + houses['parkingSpacePrice']

# We want to know if the floor is at the street level (bajo, entreplanta, semisótano, sótano) or not.
def floor_condition(floor):
    if floor == 'bj':
        return True
    elif floor == 'en':
        return True
    elif floor == 'ss':
        return True
    elif floor == 'st':
        return True
    else:
        return False
    
houses['isFloorZero'] = houses.floor.apply(floor_condition)

houses.hasLift = houses.hasLift.astype(bool)
houses.exterior = houses.exterior.astype(bool)

# Multiple Linear Regression model

houses = houses[(houses['district'] == 'Barrio de Salamanca')] # We limit our data to this district

cols = ['totalPrice', 
        'isFloorZero', 
        'size', 
        'exterior', 
        'hasLift', 
        'hasParkingSpace']
houses_model = houses[cols] # we will add the categorical variables later

houses_num = houses['size'] # selecting numerical variables
houses_cat = houses[['exterior', 'hasLift', 'hasParkingSpace', 'isFloorZero', 'propertyType', 'neighborhood']] # selecting categorical variables

sns.heatmap(houses_model.corr(numeric_only=True), annot=True)

encoded_cat = pd.get_dummies(houses_cat[['propertyType', 'neighborhood']]) # encoding categorical variables


houses_model = pd.concat([houses_model, encoded_cat], axis=1).sort_index(axis = 1) # dataframe to be used in our regression model

X = houses_model.drop('totalPrice', axis=1)
y = houses_model['totalPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

model = linear_model.LinearRegression()
result = model.fit(X_train, y_train)
prediction = result.predict(X_test)
r2_score(y_test, prediction)

# Streamlit web app
st.title("Rent price prediction")
st.write("Score of the model:")
st.write(np.round(r2_score(y_test, prediction),2))
st.sidebar.title("House Parameters")

X_pred = {}
for col in houses[['size', 'exterior', 'hasLift', 'hasParkingSpace', 'isFloorZero', 'propertyType', 'neighborhood']]:
    if col in ['size']:
        col_value = st.sidebar.slider(col, int(houses[col].min()), int(houses[col].max()), int(houses[col].mean()))
        X_pred[col] = col_value
    else:
        col_value = st.sidebar.selectbox(col, sorted(houses[col].unique()))
        X_pred[col] = col_value

X_pred = pd.DataFrame(X_pred, index=[0])
st.write("Apartment parameters:")
st.write(X_pred)

X_pred_num = X_pred['size']
X_pred_num_scaled = pd.DataFrame(X_pred_num)

X_pred_cat = X_pred[['exterior', 'hasLift', 'hasParkingSpace', 'isFloorZero', 'propertyType', 'neighborhood']]
X_pred_cat_encoded = pd.get_dummies(X_pred_cat)

cat_data = houses[['exterior', 'hasLift', 'hasParkingSpace', 'isFloorZero', 'propertyType', 'neighborhood']]
cat_data_encoded = pd.get_dummies(cat_data)
cat_encoded_cols = cat_data_encoded.columns

data_prep = pd.concat([houses_num, cat_data_encoded], axis=1).sort_index(axis = 1)
prep_columns = data_prep.columns

for col in cat_encoded_cols:
    if col not in X_pred_cat_encoded.columns:
        X_pred_cat_encoded[col] = False

X_pred_prep = pd.concat([X_pred_num_scaled, X_pred_cat_encoded], axis=1).sort_index(axis = 1)

# Predictions
pred = model.predict(X_pred_prep[prep_columns])
st.write("This is the expected rent price:")
st.write(np.round(pred))
