---
title: "Project: Prediction with Linear Regression"
excerpt: "Explore LR model <br/>"
collection: portfolio
---



This is the project from dataquest.

```python
import pandas as pd
pd.options.display.max_columns=999
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
from sklearn import linear_model

```


```python
#Read AmesHousing.tsv into a pandas data frame.
df=pd.read_csv('AmesHousing.tsv',delimiter='\t')
```

### Pre_Analysis
Create a function named train_and_test() that, for now:
1. split the data into train and test data
2. train it& test it
3. return RMSE value


```python
def transform_features(df):
    return df

def select_features(df):
    return df[['Gr Liv Area','SalePrice']]

def train_and_test(df):
    
    train=df[:1460]
    test=df[1460:]
    
    #only lfet the data are integer and float
    numeric_train=train.select_dtypes(include=['integer','float'])
    numeric_test= test.select_dtypes(include=['integer','float'])
    
    ##drop the null value
    features=train.columns.drop('SalePrice')
    lr=linear_model.LinearRegression()
    lr.fit(train[features], train['SalePrice'])
    predictions=lr.predict(test[features])
    mse=mean_squared_error(test['SalePrice'],predictions)
    rmse=np.sqrt(mse)
    
    return rmse

transform_df=transform_features(df)
filtered_df=select_features(transform_df)
rmse=train_and_test(filtered_df)
    
rmse

    
```




    57088.25161263909



# Feature Engineering 
1. remove features that we don't want to use in the model(missing, etc)
2. transform feature into the properformat

 a. (numerical to categorical, scaling numerical, filling in missing values, etc)

 b. (remove any columns that leak info about the sale(ex: the year sale))
3. create new feature by combining other features


### Handle missing value:
1. all columns: drop any with 5% or more missin gvalues for now
2. text columns: drop any with 1 or more missing 
3. numerical columns: fill in with the most common value in that column


```python
## all columns: drop any with 5% or more missing values for now
num_missing=df.isnull().sum()

##filter series to columns containing >5% missing values
drop_missing_cols=num_missing[(num_missing>len(df)/20)].sort_values()

## drop those columns from the data fram. Note the use of the .index accessor
df =df.drop(drop_missing_cols.index,axis=1)
```


```python
# text columns: drop any 1 or more missing value for now

## series object: column name -> number of missing values

text_mv_counts=df.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

## filter series to columns containg "any" missing values
drop_missing_cols_2=text_mv_counts[text_mv_counts>0]

##这个是把列中有符合某些条件的index给删了
df=df.drop(drop_missing_cols_2.index,axis=1)
```


```python
# numerical columns: fill in with the most common value in that column

##compute column-wise missing value counts
num_missing = df.select_dtypes(include=['int','float']).isnull().sum()
fixable_numeric_cols=num_missing[(num_missing<len(df)/20)& (num_missing>0)].sort_values()
fixable_numeric_cols
```




    BsmtFin SF 1       1
    BsmtFin SF 2       1
    Bsmt Unf SF        1
    Total Bsmt SF      1
    Garage Cars        1
    Garage Area        1
    Bsmt Full Bath     2
    Bsmt Half Bath     2
    Mas Vnr Area      23
    dtype: int64




```python
## compute the most common vlaue for each colum in 'fixable_nmeric_missing_cols'

##把index换成是dict，模式是‘records’
##The mode of a set of values is the value that appears most often. 
replacement_values_dict=df[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]

replacement_values_dict
```




    {'Bsmt Full Bath': 0.0,
     'Bsmt Half Bath': 0.0,
     'Bsmt Unf SF': 0.0,
     'BsmtFin SF 1': 0.0,
     'BsmtFin SF 2': 0.0,
     'Garage Area': 0.0,
     'Garage Cars': 2.0,
     'Mas Vnr Area': 0.0,
     'Total Bsmt SF': 0.0}




```python
## us pd.DataFrame.fillna() to replace missing values
df=df.fillna(replacement_values_dict)

```


```python
## verfy that very column has 0 missing values
df.isnull().sum().value_counts()
```




    0    64
    dtype: int64



### Create new feature:

1. this part can be done from the expert in this area


```python
years_sold=df['Yr Sold']-df['Year Built']
years_sold[years_sold<0]
```




    2180   -1
    dtype: int64




```python
years_since_remod=df['Yr Sold']-df['Year Remod/Add']
years_since_remod[years_since_remod<0]
```




    1702   -1
    2180   -2
    2181   -1
    dtype: int64




```python
##create new columns
df['Years Before Sale'] = years_sold
df['Years Since Remod'] = years_since_remod

##drop rows with negative values for both of these new features
df=df.drop([1702,2180,2181],axis=0)

## No longer need original year columns
df=df.drop(['Year Built','Year Remod/Add'],axis=1)
```

### Drop columns that:
1. that are not useful for ML
2. leak data about the final sale


```python
## drop columns that arenot useful for ML
df = df.drop(['PID','Order'],axis=1)

## drop columns that leak info about the final sale
df = df.drop(['Mo Sold','Sale Condition', 'Sale Type', 'Yr Sold'],axis=1)
```

### Update transform_features()

1. put previous result together


```python
def transform_features(df):
    ## all columns: drop any with 5% or more missing values for now
    num_missing=df.isnull().sum()
    ##filter series to columns containing >5% missing values
    drop_missing_cols=num_missing[(num_missing>len(df)/20)].sort_values()
    ## drop those columns from the data fram. Note the use of the .index accessor
    df =df.drop(drop_missing_cols.index,axis=1)

    ## series object: column name -> number of missing values
    text_mv_counts=df.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)
    ## filter series to columns containg "any" missing values
    drop_missing_cols_2=text_mv_counts[text_mv_counts>0]
    ##这个是把列中有符合某些条件的index给删了
    df=df.drop(drop_missing_cols_2.index,axis=1)   
    ##compute column-wise missing value counts
    num_missing = df.select_dtypes(include=['int','float']).isnull().sum()
    fixable_numeric_cols=num_missing[(num_missing<len(df)/20)& (num_missing>0)].sort_values()  
    ##The mode of a set of values is the value that appears most often. 
    replacement_values_dict=df[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
    ## us pd.DataFrame.fillna() to replace missing values
    df=df.fillna(replacement_values_dict)       

    ## create new feature
    years_sold=df['Yr Sold']-df['Year Built']
    years_sold[years_sold<0]
    years_since_remod=df['Yr Sold']-df['Year Remod/Add']
    years_since_remod[years_since_remod<0]
    ##create new columns
    df['Years Before Sale'] = years_sold
    df['Years Since Remod'] = years_since_remod
    ##drop rows with negative values for both of these new features
    df=df.drop([1702,2180,2181],axis=0)
    
    
    ## No longer need original year columns
    df=df.drop(['Year Built','Year Remod/Add'],axis=1)  
    ## drop columns that arenot useful for ML
    df = df.drop(['PID','Order'],axis=1)
    ## drop columns that leak info about the final sale
    df = df.drop(['Mo Sold','Sale Condition', 'Sale Type', 'Yr Sold'],axis=1)
    
    
    return df
```

# Feature Selection

1. Which features correlate strongly with our target column, SalePrice?
2. Which columns in the data frame should be converted to the categorical data type? 
3. Which columns are currently numerical but need to be encoded as categorical instead(because the numbers don't have any semantic meaning)?
4. What are some ways we can explore which categorical columns "correlate" well with SalePrice?
5. Update the logic for the select_features() function.


### deal with numerical features


```python
numerical_df=transform_df.select_dtypes(include=['int','float'])
numerical_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Mas Vnr Area</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>2nd Flr SF</th>
      <th>Low Qual Fin SF</th>
      <th>Gr Liv Area</th>
      <th>Bsmt Full Bath</th>
      <th>Bsmt Half Bath</th>
      <th>Full Bath</th>
      <th>Half Bath</th>
      <th>Bedroom AbvGr</th>
      <th>Kitchen AbvGr</th>
      <th>TotRms AbvGrd</th>
      <th>Fireplaces</th>
      <th>Garage Yr Blt</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>141.0</td>
      <td>31770</td>
      <td>6</td>
      <td>5</td>
      <td>1960</td>
      <td>1960</td>
      <td>112.0</td>
      <td>639.0</td>
      <td>0.0</td>
      <td>441.0</td>
      <td>1080.0</td>
      <td>1656</td>
      <td>0</td>
      <td>0</td>
      <td>1656</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>1960.0</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>210</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>80.0</td>
      <td>11622</td>
      <td>5</td>
      <td>6</td>
      <td>1961</td>
      <td>1961</td>
      <td>0.0</td>
      <td>468.0</td>
      <td>144.0</td>
      <td>270.0</td>
      <td>882.0</td>
      <td>896</td>
      <td>0</td>
      <td>0</td>
      <td>896</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1961.0</td>
      <td>1.0</td>
      <td>730.0</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>81.0</td>
      <td>14267</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>108.0</td>
      <td>923.0</td>
      <td>0.0</td>
      <td>406.0</td>
      <td>1329.0</td>
      <td>1329</td>
      <td>0</td>
      <td>0</td>
      <td>1329</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>1958.0</td>
      <td>1.0</td>
      <td>312.0</td>
      <td>393</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>93.0</td>
      <td>11160</td>
      <td>7</td>
      <td>5</td>
      <td>1968</td>
      <td>1968</td>
      <td>0.0</td>
      <td>1065.0</td>
      <td>0.0</td>
      <td>1045.0</td>
      <td>2110.0</td>
      <td>2110</td>
      <td>0</td>
      <td>0</td>
      <td>2110</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>1968.0</td>
      <td>2.0</td>
      <td>522.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>244000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>527105010</td>
      <td>60</td>
      <td>74.0</td>
      <td>13830</td>
      <td>5</td>
      <td>5</td>
      <td>1997</td>
      <td>1998</td>
      <td>0.0</td>
      <td>791.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>928.0</td>
      <td>928</td>
      <td>701</td>
      <td>0</td>
      <td>1629</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1997.0</td>
      <td>2.0</td>
      <td>482.0</td>
      <td>212</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>189900</td>
    </tr>
  </tbody>
</table>
</div>




```python
##we shall check the correlation
abs_corr_coeffs=numerical_df.corr()['SalePrice'].abs().sort_values()
abs_corr_coeffs
```




    BsmtFin SF 2       0.005891
    Misc Val           0.015691
    Yr Sold            0.030569
    Order              0.031408
    3Ssn Porch         0.032225
    Mo Sold            0.035259
    Bsmt Half Bath     0.035835
    Low Qual Fin SF    0.037660
    Pool Area          0.068403
    MS SubClass        0.085092
    Overall Cond       0.101697
    Screen Porch       0.112151
    Kitchen AbvGr      0.119814
    Enclosed Porch     0.128787
    Bedroom AbvGr      0.143913
    Bsmt Unf SF        0.182855
    PID                0.246521
    Lot Area           0.266549
    2nd Flr SF         0.269373
    Bsmt Full Bath     0.276050
    Half Bath          0.285056
    Open Porch SF      0.312951
    Wood Deck SF       0.327143
    Lot Frontage       0.357318
    BsmtFin SF 1       0.432914
    Fireplaces         0.474558
    TotRms AbvGrd      0.495474
    Mas Vnr Area       0.508285
    Garage Yr Blt      0.526965
    Year Remod/Add     0.532974
    Full Bath          0.545604
    Year Built         0.558426
    1st Flr SF         0.621676
    Total Bsmt SF      0.632280
    Garage Area        0.640401
    Garage Cars        0.647877
    Gr Liv Area        0.706780
    Overall Qual       0.799262
    SalePrice          1.000000
    Name: SalePrice, dtype: float64




```python
#select the correlation_coeffs>0.4
abs_corr_coeffs[abs_corr_coeffs>0.4]
```




    BsmtFin SF 1      0.432914
    Fireplaces        0.474558
    TotRms AbvGrd     0.495474
    Mas Vnr Area      0.508285
    Garage Yr Blt     0.526965
    Year Remod/Add    0.532974
    Full Bath         0.545604
    Year Built        0.558426
    1st Flr SF        0.621676
    Total Bsmt SF     0.632280
    Garage Area       0.640401
    Garage Cars       0.647877
    Gr Liv Area       0.706780
    Overall Qual      0.799262
    SalePrice         1.000000
    Name: SalePrice, dtype: float64




```python
#drop columns with less than 0.4 correlation with SalePrice
transform_df=transform_df.drop(abs_corr_coeffs[abs_corr_coeffs<0.4].index,axis=1)
```

### deal with categorical columns


```python
###create a list of column names form documentation that are 'meant' to be categorical

nominal_features= ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]
```

1. which columns are currently numerical but need to be encoded as categorical instead?
2. delete the categorical column has hundreds of unique values.
3. deal with dummy variable


```python
# which columns are currently numerical but need to be encoded as categorical instead?
##之前这个features是全部的feature
transform_cat_cols=[]
for col in nominal_features:
    if col in transform_df.columns:
        transform_cat_cols.append(col)
```


```python
##delete the categorical column has hundreds of unique values
uniquenss_counts=transform_df[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()
uniquenss_counts

## arbitrary cutoff of 10 unique values

drop_nonuniq_cols=uniquenss_counts[uniquenss_counts>10].index ##注意drop需要index
transform_df=transform_df.drop(drop_nonuniq_cols, axis=1)

```


```python
## select just the remianing text columns and convert to categorical
## 只有先把object换成category 才能用dummy
text_cols=transform_df.select_dtypes(include=['object'])

for col in text_cols:
    transform_df[col] = transform_df[col].astype('category')

## create dummy columns and add back to the dataframe!

transform_df=pd.concat([
        transform_df,
        pd.get_dummies(transform_df.select_dtypes(include=['category']))
        ],axis=1).drop(text_cols,axis=1)

```


```python
## update select_features()
def select_features(df, coeff_threshold=0.4, uniq_threshold=10):
    ###focus on numerical results
#     coeff_threshold=0.4
#     uniq_threshold=10
    numerical_df=df.select_dtypes(include=['int','float'])
    ##we shall check the correlation
    abs_corr_coeffs=numerical_df.corr()['SalePrice'].abs().sort_values()
    #select the correlation_coeffs>0.4
    abs_corr_coeffs[abs_corr_coeffs>coeff_threshold]
    #drop columns with less than 0.4 correlation with SalePrice
    df=df.drop(abs_corr_coeffs[abs_corr_coeffs<coeff_threshold].index,axis=1)

    # which columns are currently numerical but need to be encoded as categorical instead?
    nominal_features= ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]


    ##之前这个features是全部的feature
    transform_cat_cols=[]
    for col in nominal_features:
        if col in df.columns:
            transform_cat_cols.append(col)
    ##delete the categorical column has hundreds of unique values
    uniquenss_counts=df[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()
    ## arbitrary cutoff of 10 unique values
    drop_nonuniq_cols=uniquenss_counts[uniquenss_counts>uniq_threshold].index ##注意drop需要index
    df=df.drop(drop_nonuniq_cols, axis=1)

    ## select just the remianing text columns and convert to categorical
    ## 只有先把object换成category 才能用dummy
    text_cols=df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col] = df[col].astype('category')
    ## create dummy columns and add back to the dataframe!
    df=pd.concat([df,
            pd.get_dummies(df.select_dtypes(include=['category']))
            ],axis=1).drop(text_cols,axis=1)
    return df
```

# update the train_and_test 

1. When k equals 0, perform holdout validation (what we already implemented):
    1. Select the first 1460 rows and assign to train.
    2. Select the remaining rows and assign to test.
    3. Train on train and test on test.
    4. Compute the RMSE and return.
2. When k equals 1, perform simple cross validation:
    1. Shuffle the ordering of the rows in the data frame.
    2. Select the first 1460 rows and assign to fold_one.
    3. Select the remaining rows and assign to fold_two.
    4. Train on fold_one and test on fold_two.
    5. Train on fold_two and test on fold_one.
    6. Compute the average RMSE and return.
3. When k is greater than 0, implement k-fold cross validation using k folds:
    1. Perform k-fold cross validation using k folds.
    2. Calculate the average RMSE value and return this value.
    


```python
###let us update the k fold

def train_and_test(df,k):
    numerical_df=df.select_dtypes(include=['integer','float'])
    features=numerical_df.columns.drop('SalePrice')
    lr=linear_model.LinearRegression(0)
#     if k == 0:
#         train=df[:1460]
#         test=df[1460:]
#         lr.fit(train[features], train['SalePrice'])
#         predictions=lr.predict(test[features])
#         mse=mean_squared_error(test['SalePrice'],predictions)
#         rmse=np.sqrt(mse)
#         return rmse
    
    kf=KFold(n_splits=k, random_state=None, shuffle=False)
    rmse_values=[]
    for train_index, test_index in kf.split(df):
    ##这个df是全部的，所以要选出其feature
        train=df.iloc[train_index]
        test=df.iloc[test_index]
        lr.fit(train[features],train['SalePrice'])
        predictions=lr.predict(test[features])
        mse=mean_squared_error(test['SalePrice'],predictions)
        rmse=np.sqrt(mse)
        rmse_values.append(rmse)
    avg_rmse=np.mean(rmse_values)
    return avg_rmse
    


```

# Model Estimation


```python
df=pd.read_csv('AmesHousing.tsv',delimiter='\t')
transform_df=transform_features(df)
transform_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>Utilities</th>
      <th>Lot Config</th>
      <th>Land Slope</th>
      <th>Neighborhood</th>
      <th>Condition 1</th>
      <th>Condition 2</th>
      <th>Bldg Type</th>
      <th>House Style</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Roof Style</th>
      <th>Roof Matl</th>
      <th>Exterior 1st</th>
      <th>Exterior 2nd</th>
      <th>Mas Vnr Area</th>
      <th>Exter Qual</th>
      <th>Exter Cond</th>
      <th>Foundation</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
      <th>Heating</th>
      <th>Heating QC</th>
      <th>Central Air</th>
      <th>1st Flr SF</th>
      <th>2nd Flr SF</th>
      <th>Low Qual Fin SF</th>
      <th>Gr Liv Area</th>
      <th>Bsmt Full Bath</th>
      <th>Bsmt Half Bath</th>
      <th>Full Bath</th>
      <th>Half Bath</th>
      <th>Bedroom AbvGr</th>
      <th>Kitchen AbvGr</th>
      <th>Kitchen Qual</th>
      <th>TotRms AbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Paved Drive</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Misc Val</th>
      <th>SalePrice</th>
      <th>Years Before Sale</th>
      <th>Years Since Remod</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>RL</td>
      <td>31770</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>BrkFace</td>
      <td>Plywood</td>
      <td>112.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>639.0</td>
      <td>0.0</td>
      <td>441.0</td>
      <td>1080.0</td>
      <td>GasA</td>
      <td>Fa</td>
      <td>Y</td>
      <td>1656</td>
      <td>0</td>
      <td>0</td>
      <td>1656</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>2</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>P</td>
      <td>210</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215000</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>RH</td>
      <td>11622</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>468.0</td>
      <td>144.0</td>
      <td>270.0</td>
      <td>882.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>896</td>
      <td>0</td>
      <td>0</td>
      <td>896</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>1.0</td>
      <td>730.0</td>
      <td>Y</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>105000</td>
      <td>49</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>RL</td>
      <td>14267</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>6</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>108.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>923.0</td>
      <td>0.0</td>
      <td>406.0</td>
      <td>1329.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1329</td>
      <td>0</td>
      <td>0</td>
      <td>1329</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>1.0</td>
      <td>312.0</td>
      <td>Y</td>
      <td>393</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12500</td>
      <td>172000</td>
      <td>52</td>
      <td>52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>RL</td>
      <td>11160</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>BrkFace</td>
      <td>BrkFace</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>1065.0</td>
      <td>0.0</td>
      <td>1045.0</td>
      <td>2110.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>2110</td>
      <td>0</td>
      <td>0</td>
      <td>2110</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Ex</td>
      <td>8</td>
      <td>Typ</td>
      <td>2</td>
      <td>2.0</td>
      <td>522.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>244000</td>
      <td>42</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>RL</td>
      <td>13830</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>5</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>791.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>928.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>928</td>
      <td>701</td>
      <td>0</td>
      <td>1629</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>482.0</td>
      <td>Y</td>
      <td>212</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>189900</td>
      <td>13</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>RL</td>
      <td>9978</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>20.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>602.0</td>
      <td>0.0</td>
      <td>324.0</td>
      <td>926.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>926</td>
      <td>678</td>
      <td>0</td>
      <td>1604</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>470.0</td>
      <td>Y</td>
      <td>360</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>195500</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>120</td>
      <td>RL</td>
      <td>4920</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>StoneBr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>616.0</td>
      <td>0.0</td>
      <td>722.0</td>
      <td>1338.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1338</td>
      <td>0</td>
      <td>0</td>
      <td>1338</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>582.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>170</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>213500</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>120</td>
      <td>RL</td>
      <td>5005</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>StoneBr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>1017.0</td>
      <td>1280.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1280</td>
      <td>0</td>
      <td>0</td>
      <td>1280</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>506.0</td>
      <td>Y</td>
      <td>0</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
      <td>144</td>
      <td>0</td>
      <td>0</td>
      <td>191500</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>120</td>
      <td>RL</td>
      <td>5389</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>StoneBr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>1180.0</td>
      <td>0.0</td>
      <td>415.0</td>
      <td>1595.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1616</td>
      <td>0</td>
      <td>0</td>
      <td>1616</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>Y</td>
      <td>237</td>
      <td>152</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>236500</td>
      <td>15</td>
      <td>14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>60</td>
      <td>RL</td>
      <td>7500</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>994.0</td>
      <td>994.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>1028</td>
      <td>776</td>
      <td>0</td>
      <td>1804</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>442.0</td>
      <td>Y</td>
      <td>140</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>189000</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>10</th>
      <td>60</td>
      <td>RL</td>
      <td>10000</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>763.0</td>
      <td>763.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>763</td>
      <td>892</td>
      <td>0</td>
      <td>1655</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>440.0</td>
      <td>Y</td>
      <td>157</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>175900</td>
      <td>17</td>
      <td>16</td>
    </tr>
    <tr>
      <th>11</th>
      <td>20</td>
      <td>RL</td>
      <td>7980</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>7</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>0.0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>PConc</td>
      <td>935.0</td>
      <td>0.0</td>
      <td>233.0</td>
      <td>1168.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1187</td>
      <td>0</td>
      <td>0</td>
      <td>1187</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>420.0</td>
      <td>Y</td>
      <td>483</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>500</td>
      <td>185000</td>
      <td>18</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>60</td>
      <td>RL</td>
      <td>8402</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>789.0</td>
      <td>789.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>789</td>
      <td>676</td>
      <td>0</td>
      <td>1465</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>393.0</td>
      <td>Y</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>180400</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20</td>
      <td>RL</td>
      <td>10176</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>637.0</td>
      <td>0.0</td>
      <td>663.0</td>
      <td>1300.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>1341</td>
      <td>0</td>
      <td>0</td>
      <td>1341</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>506.0</td>
      <td>Y</td>
      <td>192</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>171500</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>14</th>
      <td>120</td>
      <td>RL</td>
      <td>6820</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>StoneBr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>368.0</td>
      <td>1120.0</td>
      <td>0.0</td>
      <td>1488.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1502</td>
      <td>0</td>
      <td>0</td>
      <td>1502</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Gd</td>
      <td>4</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>Y</td>
      <td>0</td>
      <td>54</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>212000</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <th>15</th>
      <td>60</td>
      <td>RL</td>
      <td>53504</td>
      <td>Pave</td>
      <td>IR2</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>Mod</td>
      <td>StoneBr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>Wd Shng</td>
      <td>603.0</td>
      <td>Ex</td>
      <td>TA</td>
      <td>PConc</td>
      <td>1416.0</td>
      <td>0.0</td>
      <td>234.0</td>
      <td>1650.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1690</td>
      <td>1589</td>
      <td>0</td>
      <td>3279</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Ex</td>
      <td>12</td>
      <td>Mod</td>
      <td>1</td>
      <td>3.0</td>
      <td>841.0</td>
      <td>Y</td>
      <td>503</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>210</td>
      <td>0</td>
      <td>0</td>
      <td>538000</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>16</th>
      <td>50</td>
      <td>RL</td>
      <td>12134</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Bnk</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>8</td>
      <td>7</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Wood</td>
      <td>427.0</td>
      <td>0.0</td>
      <td>132.0</td>
      <td>559.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>1080</td>
      <td>672</td>
      <td>0</td>
      <td>1752</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>492.0</td>
      <td>Y</td>
      <td>325</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>164000</td>
      <td>22</td>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>20</td>
      <td>RL</td>
      <td>11394</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>StoneBr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>9</td>
      <td>2</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>1445.0</td>
      <td>0.0</td>
      <td>411.0</td>
      <td>1856.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1856</td>
      <td>0</td>
      <td>0</td>
      <td>1856</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Ex</td>
      <td>8</td>
      <td>Typ</td>
      <td>1</td>
      <td>3.0</td>
      <td>834.0</td>
      <td>Y</td>
      <td>113</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>394432</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20</td>
      <td>RL</td>
      <td>19138</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>744.0</td>
      <td>864.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>864</td>
      <td>0</td>
      <td>0</td>
      <td>864</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>4</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>400.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>141000</td>
      <td>59</td>
      <td>59</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>RL</td>
      <td>13175</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NWAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>119.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>790.0</td>
      <td>163.0</td>
      <td>589.0</td>
      <td>1542.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>2073</td>
      <td>0</td>
      <td>0</td>
      <td>2073</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Min1</td>
      <td>2</td>
      <td>2.0</td>
      <td>500.0</td>
      <td>Y</td>
      <td>349</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>210000</td>
      <td>32</td>
      <td>22</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>RL</td>
      <td>11751</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NWAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>6</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>480.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>705.0</td>
      <td>0.0</td>
      <td>1139.0</td>
      <td>1844.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1844</td>
      <td>0</td>
      <td>0</td>
      <td>1844</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>546.0</td>
      <td>Y</td>
      <td>0</td>
      <td>122</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>190000</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>21</th>
      <td>85</td>
      <td>RL</td>
      <td>10625</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NWAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>SFoyer</td>
      <td>7</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>81.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>885.0</td>
      <td>168.0</td>
      <td>0.0</td>
      <td>1053.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1173</td>
      <td>0</td>
      <td>0</td>
      <td>1173</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>2</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>Y</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>170000</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>22</th>
      <td>60</td>
      <td>FV</td>
      <td>7500</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Somerst</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>533.0</td>
      <td>0.0</td>
      <td>281.0</td>
      <td>814.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>814</td>
      <td>860</td>
      <td>0</td>
      <td>1674</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>663.0</td>
      <td>Y</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>216000</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>23</th>
      <td>20</td>
      <td>RL</td>
      <td>11241</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>7</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>180.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>578.0</td>
      <td>0.0</td>
      <td>426.0</td>
      <td>1004.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1004</td>
      <td>0</td>
      <td>0</td>
      <td>1004</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>480.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>700</td>
      <td>149000</td>
      <td>40</td>
      <td>40</td>
    </tr>
    <tr>
      <th>24</th>
      <td>20</td>
      <td>RL</td>
      <td>12537</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>734.0</td>
      <td>0.0</td>
      <td>344.0</td>
      <td>1078.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1078</td>
      <td>0</td>
      <td>0</td>
      <td>1078</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>500.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>149900</td>
      <td>39</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>20</td>
      <td>RL</td>
      <td>8450</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>775.0</td>
      <td>0.0</td>
      <td>281.0</td>
      <td>1056.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1056</td>
      <td>0</td>
      <td>0</td>
      <td>1056</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>1.0</td>
      <td>304.0</td>
      <td>Y</td>
      <td>0</td>
      <td>85</td>
      <td>184</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142000</td>
      <td>42</td>
      <td>42</td>
    </tr>
    <tr>
      <th>26</th>
      <td>20</td>
      <td>RL</td>
      <td>8400</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>804.0</td>
      <td>78.0</td>
      <td>0.0</td>
      <td>882.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>882</td>
      <td>0</td>
      <td>0</td>
      <td>882</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>4</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>525.0</td>
      <td>Y</td>
      <td>240</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>126000</td>
      <td>40</td>
      <td>40</td>
    </tr>
    <tr>
      <th>27</th>
      <td>20</td>
      <td>RL</td>
      <td>10500</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>432.0</td>
      <td>0.0</td>
      <td>432.0</td>
      <td>864.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>864</td>
      <td>0</td>
      <td>0</td>
      <td>864</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>115000</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>28</th>
      <td>120</td>
      <td>RH</td>
      <td>5858</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>1051.0</td>
      <td>0.0</td>
      <td>354.0</td>
      <td>1405.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1337</td>
      <td>0</td>
      <td>0</td>
      <td>1337</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>511.0</td>
      <td>Y</td>
      <td>203</td>
      <td>68</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>184000</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>29</th>
      <td>160</td>
      <td>RM</td>
      <td>1680</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>BrDale</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>2Story</td>
      <td>6</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>504.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>327.0</td>
      <td>483.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>483</td>
      <td>504</td>
      <td>0</td>
      <td>987</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>1.0</td>
      <td>264.0</td>
      <td>Y</td>
      <td>275</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96000</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2900</th>
      <td>20</td>
      <td>RL</td>
      <td>13618</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Timber</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>198.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>1350.0</td>
      <td>0.0</td>
      <td>378.0</td>
      <td>1728.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1960</td>
      <td>0</td>
      <td>0</td>
      <td>1960</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>2</td>
      <td>3.0</td>
      <td>714.0</td>
      <td>Y</td>
      <td>172</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>320000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2901</th>
      <td>20</td>
      <td>RL</td>
      <td>11443</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Timber</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>208.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>1460.0</td>
      <td>0.0</td>
      <td>408.0</td>
      <td>1868.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>2028</td>
      <td>0</td>
      <td>0</td>
      <td>2028</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>2</td>
      <td>3.0</td>
      <td>880.0</td>
      <td>Y</td>
      <td>326</td>
      <td>66</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>369900</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2902</th>
      <td>20</td>
      <td>RL</td>
      <td>11577</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Timber</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>9</td>
      <td>5</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>382.0</td>
      <td>Ex</td>
      <td>TA</td>
      <td>PConc</td>
      <td>1455.0</td>
      <td>0.0</td>
      <td>383.0</td>
      <td>1838.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1838</td>
      <td>0</td>
      <td>0</td>
      <td>1838</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Ex</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>3.0</td>
      <td>682.0</td>
      <td>Y</td>
      <td>161</td>
      <td>225</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>359900</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2903</th>
      <td>20</td>
      <td>A (agr)</td>
      <td>31250</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Artery</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>1</td>
      <td>3</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CBlock</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>Fa</td>
      <td>CBlock</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1600</td>
      <td>0</td>
      <td>0</td>
      <td>1600</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Mod</td>
      <td>0</td>
      <td>1.0</td>
      <td>270.0</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>135</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>81500</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>2904</th>
      <td>90</td>
      <td>RM</td>
      <td>7020</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>SFoyer</td>
      <td>7</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>200.0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>PConc</td>
      <td>1243.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>1288.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>1368</td>
      <td>0</td>
      <td>0</td>
      <td>1368</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>TA</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>4.0</td>
      <td>784.0</td>
      <td>Y</td>
      <td>0</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215000</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2905</th>
      <td>120</td>
      <td>RM</td>
      <td>4500</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>116.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>897.0</td>
      <td>0.0</td>
      <td>319.0</td>
      <td>1216.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1216</td>
      <td>0</td>
      <td>0</td>
      <td>1216</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>402.0</td>
      <td>Y</td>
      <td>0</td>
      <td>125</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>164000</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2906</th>
      <td>120</td>
      <td>RM</td>
      <td>4500</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>443.0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>PConc</td>
      <td>1201.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>1237.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1337</td>
      <td>0</td>
      <td>0</td>
      <td>1337</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>405.0</td>
      <td>Y</td>
      <td>0</td>
      <td>199</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>153500</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2907</th>
      <td>20</td>
      <td>RL</td>
      <td>17217</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1140.0</td>
      <td>1140.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1140</td>
      <td>0</td>
      <td>0</td>
      <td>1140</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Y</td>
      <td>36</td>
      <td>56</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>84500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>160</td>
      <td>RM</td>
      <td>2665</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>2Story</td>
      <td>5</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>264.0</td>
      <td>264.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>616</td>
      <td>688</td>
      <td>0</td>
      <td>1304</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>1.0</td>
      <td>336.0</td>
      <td>Y</td>
      <td>141</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>104500</td>
      <td>29</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2909</th>
      <td>160</td>
      <td>RM</td>
      <td>2665</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>2Story</td>
      <td>5</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>548.0</td>
      <td>173.0</td>
      <td>36.0</td>
      <td>757.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>925</td>
      <td>550</td>
      <td>0</td>
      <td>1475</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>1.0</td>
      <td>336.0</td>
      <td>Y</td>
      <td>104</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>127000</td>
      <td>29</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2910</th>
      <td>160</td>
      <td>RM</td>
      <td>3964</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>2Story</td>
      <td>6</td>
      <td>4</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>837.0</td>
      <td>0.0</td>
      <td>105.0</td>
      <td>942.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>1291</td>
      <td>1230</td>
      <td>0</td>
      <td>2521</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>TA</td>
      <td>10</td>
      <td>Maj1</td>
      <td>1</td>
      <td>2.0</td>
      <td>576.0</td>
      <td>Y</td>
      <td>728</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>151400</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2911</th>
      <td>20</td>
      <td>RL</td>
      <td>10172</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>7</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>441.0</td>
      <td>0.0</td>
      <td>423.0</td>
      <td>864.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>874</td>
      <td>0</td>
      <td>0</td>
      <td>874</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>1.0</td>
      <td>288.0</td>
      <td>Y</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>126500</td>
      <td>38</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2912</th>
      <td>90</td>
      <td>RL</td>
      <td>11836</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>149.0</td>
      <td>0.0</td>
      <td>1503.0</td>
      <td>1652.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1652</td>
      <td>0</td>
      <td>0</td>
      <td>1652</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>TA</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>3.0</td>
      <td>928.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>146500</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2913</th>
      <td>180</td>
      <td>RM</td>
      <td>1470</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>SFoyer</td>
      <td>4</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>522.0</td>
      <td>0.0</td>
      <td>108.0</td>
      <td>630.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>630</td>
      <td>0</td>
      <td>0</td>
      <td>630</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>TA</td>
      <td>3</td>
      <td>Typ</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>73000</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2914</th>
      <td>160</td>
      <td>RM</td>
      <td>1484</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>2Story</td>
      <td>4</td>
      <td>4</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>252.0</td>
      <td>0.0</td>
      <td>294.0</td>
      <td>546.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>1.0</td>
      <td>253.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>79400</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2915</th>
      <td>20</td>
      <td>RL</td>
      <td>13384</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>194.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>119.0</td>
      <td>344.0</td>
      <td>641.0</td>
      <td>1104.0</td>
      <td>GasA</td>
      <td>Fa</td>
      <td>Y</td>
      <td>1360</td>
      <td>0</td>
      <td>0</td>
      <td>1360</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>Typ</td>
      <td>1</td>
      <td>1.0</td>
      <td>336.0</td>
      <td>Y</td>
      <td>160</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>140000</td>
      <td>37</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2916</th>
      <td>180</td>
      <td>RM</td>
      <td>1533</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>SFoyer</td>
      <td>5</td>
      <td>7</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>553.0</td>
      <td>0.0</td>
      <td>77.0</td>
      <td>630.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>630</td>
      <td>0</td>
      <td>0</td>
      <td>630</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Ex</td>
      <td>3</td>
      <td>Typ</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>92000</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2917</th>
      <td>160</td>
      <td>RM</td>
      <td>1533</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>2Story</td>
      <td>4</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>408.0</td>
      <td>0.0</td>
      <td>138.0</td>
      <td>546.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>1.0</td>
      <td>286.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>87550</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2918</th>
      <td>160</td>
      <td>RM</td>
      <td>1526</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>2Story</td>
      <td>4</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>546.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Y</td>
      <td>0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>79500</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2919</th>
      <td>160</td>
      <td>RM</td>
      <td>1936</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>2Story</td>
      <td>4</td>
      <td>7</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>546.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>90500</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2920</th>
      <td>160</td>
      <td>RM</td>
      <td>1894</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>2Story</td>
      <td>4</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>252.0</td>
      <td>0.0</td>
      <td>294.0</td>
      <td>546.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>1.0</td>
      <td>286.0</td>
      <td>Y</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>71000</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2921</th>
      <td>90</td>
      <td>RL</td>
      <td>12640</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>936.0</td>
      <td>396.0</td>
      <td>396.0</td>
      <td>1728.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1728</td>
      <td>0</td>
      <td>0</td>
      <td>1728</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>TA</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>574.0</td>
      <td>Y</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>150900</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2922</th>
      <td>90</td>
      <td>RL</td>
      <td>9297</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>1606.0</td>
      <td>0.0</td>
      <td>122.0</td>
      <td>1728.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1728</td>
      <td>0</td>
      <td>0</td>
      <td>1728</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>TA</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>560.0</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>188000</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>20</td>
      <td>RL</td>
      <td>17400</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Low</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>BrkFace</td>
      <td>BrkFace</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>936.0</td>
      <td>0.0</td>
      <td>190.0</td>
      <td>1126.0</td>
      <td>GasA</td>
      <td>Fa</td>
      <td>Y</td>
      <td>1126</td>
      <td>0</td>
      <td>0</td>
      <td>1126</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>484.0</td>
      <td>P</td>
      <td>295</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>160000</td>
      <td>29</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2924</th>
      <td>20</td>
      <td>RL</td>
      <td>20000</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>7</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>1224.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1224.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>1224</td>
      <td>0</td>
      <td>0</td>
      <td>1224</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>576.0</td>
      <td>Y</td>
      <td>474</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>131000</td>
      <td>46</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2925</th>
      <td>80</td>
      <td>RL</td>
      <td>7937</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>SLvl</td>
      <td>6</td>
      <td>6</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>819.0</td>
      <td>0.0</td>
      <td>184.0</td>
      <td>1003.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1003</td>
      <td>0</td>
      <td>0</td>
      <td>1003</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>588.0</td>
      <td>Y</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142500</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>20</td>
      <td>RL</td>
      <td>8885</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Low</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>301.0</td>
      <td>324.0</td>
      <td>239.0</td>
      <td>864.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>902</td>
      <td>0</td>
      <td>0</td>
      <td>902</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>2.0</td>
      <td>484.0</td>
      <td>Y</td>
      <td>164</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>131000</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>85</td>
      <td>RL</td>
      <td>10441</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>SFoyer</td>
      <td>5</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>Wd Shng</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>337.0</td>
      <td>0.0</td>
      <td>575.0</td>
      <td>912.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>970</td>
      <td>0</td>
      <td>0</td>
      <td>970</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Y</td>
      <td>80</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>700</td>
      <td>132000</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>20</td>
      <td>RL</td>
      <td>10010</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>1071.0</td>
      <td>123.0</td>
      <td>195.0</td>
      <td>1389.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>1389</td>
      <td>0</td>
      <td>0</td>
      <td>1389</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>2.0</td>
      <td>418.0</td>
      <td>Y</td>
      <td>240</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>170000</td>
      <td>32</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>60</td>
      <td>RL</td>
      <td>9627</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>94.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>758.0</td>
      <td>0.0</td>
      <td>238.0</td>
      <td>996.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>996</td>
      <td>1004</td>
      <td>0</td>
      <td>2000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>3.0</td>
      <td>650.0</td>
      <td>Y</td>
      <td>190</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>188000</td>
      <td>13</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>2927 rows × 58 columns</p>
</div>




```python
filtered_df=select_features(transform_df,coeff_threshold=0.4, uniq_threshold=10)
rmse=train_and_test(filtered_df,k=10)
```


```python
rmse
```




    28754.407162175525




```python

```
