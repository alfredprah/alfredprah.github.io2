---
title: '1. Bracketology'
subtitle: 'Which college basketball teams will make it to the NCAA tournament this year?'
date: 2018-06-30 00:00:00
featured_image: '/images/demo/basketball-885786_1920.jpg'
excerpt: The purpose of this project is to predict which college teams will make it to the annual NCAA men's basketball tournament this year. 
---

![](/images/basketball-college.jpg)


#### The NCAA Division I Men's Basketball Tournament, also known and branded as NCAA March Madness, is a single-elimination tournament played each spring in the United States, currently featuring 68 college basketball teams from the Division I level of the National Collegiate Athletic Association (NCAA), to determine the national championship. The tournament was created in 1939 by the National Association of Basketball Coaches, and was the idea of Ohio State coach Harold Olsen. Played mostly during March, it has become one of the most famous annual sporting events in the United States. In this project, I walk you through the methodology I employed to build a predictive model with a 94% accuracy score (AUC). 

#### This file, 50-master, is a combination of 4 notebooks: 10-import, 20-Exploratory_Data_Analysis, 30-Feature_Engineering, 40-Modeling (from my Github)

### 10-import

#### Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
```

#### Read in data


```python
df = pd.read_csv('cbb.csv')
```

#### Inspect data


```python
## view the first 5 rows of the dataset
df.head()
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
      <th>TEAM</th>
      <th>CONF</th>
      <th>G</th>
      <th>W</th>
      <th>ADJOE</th>
      <th>ADJDE</th>
      <th>BARTHAG</th>
      <th>EFG_O</th>
      <th>EFG_D</th>
      <th>TOR</th>
      <th>...</th>
      <th>FTRD</th>
      <th>2P_O</th>
      <th>2P_D</th>
      <th>3P_O</th>
      <th>3P_D</th>
      <th>ADJ_T</th>
      <th>WAB</th>
      <th>POSTSEASON</th>
      <th>SEED</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>North Carolina</td>
      <td>ACC</td>
      <td>40</td>
      <td>33</td>
      <td>123.3</td>
      <td>94.9</td>
      <td>0.9531</td>
      <td>52.6</td>
      <td>48.1</td>
      <td>15.4</td>
      <td>...</td>
      <td>30.4</td>
      <td>53.9</td>
      <td>44.6</td>
      <td>32.7</td>
      <td>36.2</td>
      <td>71.7</td>
      <td>8.6</td>
      <td>2ND</td>
      <td>1.0</td>
      <td>2016</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Wisconsin</td>
      <td>B10</td>
      <td>40</td>
      <td>36</td>
      <td>129.1</td>
      <td>93.6</td>
      <td>0.9758</td>
      <td>54.8</td>
      <td>47.7</td>
      <td>12.4</td>
      <td>...</td>
      <td>22.4</td>
      <td>54.8</td>
      <td>44.7</td>
      <td>36.5</td>
      <td>37.5</td>
      <td>59.3</td>
      <td>11.3</td>
      <td>2ND</td>
      <td>1.0</td>
      <td>2015</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Michigan</td>
      <td>B10</td>
      <td>40</td>
      <td>33</td>
      <td>114.4</td>
      <td>90.4</td>
      <td>0.9375</td>
      <td>53.9</td>
      <td>47.7</td>
      <td>14.0</td>
      <td>...</td>
      <td>30.0</td>
      <td>54.7</td>
      <td>46.8</td>
      <td>35.2</td>
      <td>33.2</td>
      <td>65.9</td>
      <td>6.9</td>
      <td>2ND</td>
      <td>3.0</td>
      <td>2018</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Texas Tech</td>
      <td>B12</td>
      <td>38</td>
      <td>31</td>
      <td>115.2</td>
      <td>85.2</td>
      <td>0.9696</td>
      <td>53.5</td>
      <td>43.0</td>
      <td>17.7</td>
      <td>...</td>
      <td>36.6</td>
      <td>52.8</td>
      <td>41.9</td>
      <td>36.5</td>
      <td>29.7</td>
      <td>67.5</td>
      <td>7.0</td>
      <td>2ND</td>
      <td>3.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gonzaga</td>
      <td>WCC</td>
      <td>39</td>
      <td>37</td>
      <td>117.8</td>
      <td>86.3</td>
      <td>0.9728</td>
      <td>56.6</td>
      <td>41.1</td>
      <td>16.2</td>
      <td>...</td>
      <td>26.9</td>
      <td>56.3</td>
      <td>40.0</td>
      <td>38.2</td>
      <td>29.0</td>
      <td>71.5</td>
      <td>7.7</td>
      <td>2ND</td>
      <td>1.0</td>
      <td>2017</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1757 entries, 0 to 1756
    Data columns (total 24 columns):
    TEAM          1757 non-null object
    CONF          1757 non-null object
    G             1757 non-null int64
    W             1757 non-null int64
    ADJOE         1757 non-null float64
    ADJDE         1757 non-null float64
    BARTHAG       1757 non-null float64
    EFG_O         1757 non-null float64
    EFG_D         1757 non-null float64
    TOR           1757 non-null float64
    TORD          1757 non-null float64
    ORB           1757 non-null float64
    DRB           1757 non-null float64
    FTR           1757 non-null float64
    FTRD          1757 non-null float64
    2P_O          1757 non-null float64
    2P_D          1757 non-null float64
    3P_O          1757 non-null float64
    3P_D          1757 non-null float64
    ADJ_T         1757 non-null float64
    WAB           1757 non-null float64
    POSTSEASON    340 non-null object
    SEED          340 non-null float64
    YEAR          1757 non-null int64
    dtypes: float64(18), int64(3), object(3)
    memory usage: 329.6+ KB



```python
df.describe()
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
      <th>G</th>
      <th>W</th>
      <th>ADJOE</th>
      <th>ADJDE</th>
      <th>BARTHAG</th>
      <th>EFG_O</th>
      <th>EFG_D</th>
      <th>TOR</th>
      <th>TORD</th>
      <th>ORB</th>
      <th>...</th>
      <th>FTR</th>
      <th>FTRD</th>
      <th>2P_O</th>
      <th>2P_D</th>
      <th>3P_O</th>
      <th>3P_D</th>
      <th>ADJ_T</th>
      <th>WAB</th>
      <th>SEED</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>...</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>1757.000000</td>
      <td>340.000000</td>
      <td>1757.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>31.523051</td>
      <td>16.513375</td>
      <td>103.542402</td>
      <td>103.542459</td>
      <td>0.493398</td>
      <td>50.120489</td>
      <td>50.312806</td>
      <td>18.591804</td>
      <td>18.521286</td>
      <td>29.277120</td>
      <td>...</td>
      <td>35.097894</td>
      <td>35.373307</td>
      <td>49.135970</td>
      <td>49.298065</td>
      <td>34.563517</td>
      <td>34.744792</td>
      <td>68.422254</td>
      <td>-7.837109</td>
      <td>8.791176</td>
      <td>2017.002277</td>
    </tr>
    <tr>
      <td>std</td>
      <td>2.602819</td>
      <td>6.545571</td>
      <td>7.304975</td>
      <td>6.472676</td>
      <td>0.255291</td>
      <td>3.130430</td>
      <td>2.859604</td>
      <td>1.991637</td>
      <td>2.108968</td>
      <td>4.101782</td>
      <td>...</td>
      <td>4.884599</td>
      <td>5.900935</td>
      <td>3.422136</td>
      <td>3.288265</td>
      <td>2.742323</td>
      <td>2.369727</td>
      <td>3.258920</td>
      <td>6.988694</td>
      <td>4.674090</td>
      <td>1.415419</td>
    </tr>
    <tr>
      <td>min</td>
      <td>24.000000</td>
      <td>0.000000</td>
      <td>76.700000</td>
      <td>84.000000</td>
      <td>0.007700</td>
      <td>39.400000</td>
      <td>39.600000</td>
      <td>12.400000</td>
      <td>10.200000</td>
      <td>15.000000</td>
      <td>...</td>
      <td>21.600000</td>
      <td>21.800000</td>
      <td>37.700000</td>
      <td>37.700000</td>
      <td>25.200000</td>
      <td>27.100000</td>
      <td>57.200000</td>
      <td>-25.200000</td>
      <td>1.000000</td>
      <td>2015.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>30.000000</td>
      <td>12.000000</td>
      <td>98.600000</td>
      <td>98.900000</td>
      <td>0.283700</td>
      <td>48.100000</td>
      <td>48.400000</td>
      <td>17.200000</td>
      <td>17.100000</td>
      <td>26.600000</td>
      <td>...</td>
      <td>31.700000</td>
      <td>31.200000</td>
      <td>46.900000</td>
      <td>47.100000</td>
      <td>32.600000</td>
      <td>33.100000</td>
      <td>66.400000</td>
      <td>-13.000000</td>
      <td>5.000000</td>
      <td>2016.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>31.000000</td>
      <td>16.000000</td>
      <td>103.100000</td>
      <td>103.800000</td>
      <td>0.474000</td>
      <td>50.000000</td>
      <td>50.300000</td>
      <td>18.500000</td>
      <td>18.500000</td>
      <td>29.400000</td>
      <td>...</td>
      <td>34.900000</td>
      <td>34.900000</td>
      <td>49.000000</td>
      <td>49.300000</td>
      <td>34.600000</td>
      <td>34.700000</td>
      <td>68.500000</td>
      <td>-8.400000</td>
      <td>9.000000</td>
      <td>2017.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>33.000000</td>
      <td>21.000000</td>
      <td>108.100000</td>
      <td>108.000000</td>
      <td>0.710600</td>
      <td>52.100000</td>
      <td>52.300000</td>
      <td>19.800000</td>
      <td>19.900000</td>
      <td>31.900000</td>
      <td>...</td>
      <td>38.300000</td>
      <td>39.200000</td>
      <td>51.400000</td>
      <td>51.500000</td>
      <td>36.400000</td>
      <td>36.300000</td>
      <td>70.400000</td>
      <td>-3.100000</td>
      <td>13.000000</td>
      <td>2018.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>40.000000</td>
      <td>38.000000</td>
      <td>129.100000</td>
      <td>124.000000</td>
      <td>0.984200</td>
      <td>59.800000</td>
      <td>59.500000</td>
      <td>26.100000</td>
      <td>28.000000</td>
      <td>42.100000</td>
      <td>...</td>
      <td>51.000000</td>
      <td>58.500000</td>
      <td>62.600000</td>
      <td>61.200000</td>
      <td>44.100000</td>
      <td>43.100000</td>
      <td>83.400000</td>
      <td>13.100000</td>
      <td>16.000000</td>
      <td>2019.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>




```python
## size of dataset
df.shape
```




    (1757, 24)




```python
## any NA values
df.isnull().sum()
```




    TEAM             0
    CONF             0
    G                0
    W                0
    ADJOE            0
    ADJDE            0
    BARTHAG          0
    EFG_O            0
    EFG_D            0
    TOR              0
    TORD             0
    ORB              0
    DRB              0
    FTR              0
    FTRD             0
    2P_O             0
    2P_D             0
    3P_O             0
    3P_D             0
    ADJ_T            0
    WAB              0
    POSTSEASON    1417
    SEED          1417
    YEAR             0
    dtype: int64



##### From this, I can conclude/confirm that there are only two columns that contain NA values. In the case of this dataset and for the purposes of this project, these NA values actually tell us something: that "(1757-340)= 1417" teams have never made it to the March Madness tournament.

 

#### Inspecting the target column, "SEED"


```python
df['SEED'].values
```




    array([ 1.,  1.,  3., ...,  2., 11.,  4.])




```python
df['SEED'].value_counts
```




    <bound method IndexOpsMixin.value_counts of 0        1.0
    1        1.0
    2        3.0
    3        3.0
    4        1.0
            ... 
    1752     7.0
    1753     3.0
    1754     2.0
    1755    11.0
    1756     4.0
    Name: SEED, Length: 1757, dtype: float64>




```python
print(np.min(df['SEED']))
print(np.max(df['SEED']))
```

    1.0
    16.0


##### This informs me that the highest SEED number is 16, for each basketball tournament. (Refer to *00_dataset-variables.ipynb* for more information on "SEED")

 


```python
df['YEAR'].unique()
```




    array([2016, 2015, 2018, 2019, 2017])



##### The dataset spans 5 years.

 

### 20-Exploratory Data Analysis

#### Which columns in the dataset contain numeric values?


```python
numeric_df = df.select_dtypes(include=['int', 'float'])

# Print the column names contained in df
print(numeric_df.columns)
```

    Index(['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD',
           'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T',
           'WAB', 'SEED', 'YEAR'],
          dtype='object')


#### Explore the values in each of these numeric columns to determine which ones are necessary for the purposes of my project:


```python
print(df['G'].values)
print(df['W'].values)
print(df['ADJOE'].values)
print(df['BARTHAG'].values)
print(df['EFG_O'].values)
```

    [40 40 40 ... 36 35 37]
    [33 36 33 ... 31 27 32]
    [123.3 129.1 114.4 ... 122.8 117.4 117.2]
    [0.9531 0.9758 0.9375 ... 0.9488 0.9238 0.9192]
    [52.6 54.8 53.9 ... 55.3 55.2 57. ]


##### From inspecting the values of only 5 columns fromthe dataset, I can tell that there is the need to either standardize or normalize the values in the various numeric columns of my dataset. I will do this in the feature emginnering notebook, "30-Feature Engineering"

 

#### Which columns in the dataset contain string values?


```python
string_df = df.select_dtypes(include=['object'])
print(string_df.columns)
```

    Index(['TEAM', 'CONF', 'POSTSEASON'], dtype='object')


#### Explore the values in each of these three columns to determine if they should be used for predictions:


```python
print(df['TEAM'].values)
print(df['CONF'].values)
print(df['POSTSEASON'].values)
```

    ['North Carolina' 'Wisconsin' 'Michigan' ... 'Tennessee' 'Gonzaga'
     'Gonzaga']
    ['ACC' 'B10' 'B10' ... 'SEC' 'WCC' 'WCC']
    ['2ND' '2ND' '2ND' ... 'S16' 'S16' 'S16']


##### It could be helpful to explore the number of teams that make it to the tournament from each league (Ex. ACC, B10, SEC, etc), how highly they're ranked in the national tournament, and how their performaces vary from year to year. However, these explorations would not be necessary, for the purposes of my project. 

 

#### Check the distribution of the features in the dataset because most ML models assume that the data is normally distributed:


```python
df.hist()
plt.show()
```


![png](output_38_0.png)


#### I'd like to explore the possibility of some of the features in the dataset being correlated/having a relationship:


```python
sns.pairplot(df)
plt.show()
```


![png](output_40_0.png)


#### From quick observation, some of the positive correlations make a lot of sense. For example:
- BARTHAG, Power Rating & W, Number of games won. A team with a higher chance of beating an average Division I team can be expected to have a high number of wins. 

- 2P_D, Two-Point Shooting Percentage Allowed & ADJDE, Adjusted Defensive Efficiency. A team that prevents the opposing team from making baskets can be said to be efficient, defensively. 

 

### 30-Feature Engineering 

#### label the target column, 'Y'


```python
Y = df['SEED'].copy()
```

#### Repalce the null values with 0 since it was confirmed that the null values in the dataset are non-trivial: they represent teams that never made it to the basketball tournament


```python
Y.fillna(0, inplace = True)
```


```python
Y.value_counts()
```




    0.0     1417
    16.0      30
    11.0      30
    3.0       21
    15.0      20
    10.0      20
    2.0       20
    6.0       20
    9.0       20
    7.0       20
    14.0      20
    5.0       20
    8.0       20
    12.0      20
    13.0      20
    1.0       20
    4.0       19
    Name: SEED, dtype: int64



#### Replace values ranging from 1-16 with '1' so that '1' will refer to teams that made it to the tournament and '0' will represent teams that did not make it


```python
Y.replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 1, inplace=True)
```

#### Remove string columns since it was determined in "20-Exploratory_Data_Analysis" that these columns will not be used for the purposes of this project, as well as the target column, 'SEED'/'Y


```python
string_df
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
      <th>TEAM</th>
      <th>CONF</th>
      <th>POSTSEASON</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>North Carolina</td>
      <td>ACC</td>
      <td>2ND</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Wisconsin</td>
      <td>B10</td>
      <td>2ND</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Michigan</td>
      <td>B10</td>
      <td>2ND</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Texas Tech</td>
      <td>B12</td>
      <td>2ND</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gonzaga</td>
      <td>WCC</td>
      <td>2ND</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1752</td>
      <td>Texas A&amp;M</td>
      <td>SEC</td>
      <td>S16</td>
    </tr>
    <tr>
      <td>1753</td>
      <td>LSU</td>
      <td>SEC</td>
      <td>S16</td>
    </tr>
    <tr>
      <td>1754</td>
      <td>Tennessee</td>
      <td>SEC</td>
      <td>S16</td>
    </tr>
    <tr>
      <td>1755</td>
      <td>Gonzaga</td>
      <td>WCC</td>
      <td>S16</td>
    </tr>
    <tr>
      <td>1756</td>
      <td>Gonzaga</td>
      <td>WCC</td>
      <td>S16</td>
    </tr>
  </tbody>
</table>
<p>1757 rows × 3 columns</p>
</div>




```python
strings = ['TEAM', 'CONF', 'POSTSEASON', 'SEED']
df.drop(columns = strings, inplace = True)
```


```python
df.columns
```




    Index(['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD',
           'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T',
           'WAB', 'YEAR'],
          dtype='object')




```python
df
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
      <th>G</th>
      <th>W</th>
      <th>ADJOE</th>
      <th>ADJDE</th>
      <th>BARTHAG</th>
      <th>EFG_O</th>
      <th>EFG_D</th>
      <th>TOR</th>
      <th>TORD</th>
      <th>ORB</th>
      <th>DRB</th>
      <th>FTR</th>
      <th>FTRD</th>
      <th>2P_O</th>
      <th>2P_D</th>
      <th>3P_O</th>
      <th>3P_D</th>
      <th>ADJ_T</th>
      <th>WAB</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>40</td>
      <td>33</td>
      <td>123.3</td>
      <td>94.9</td>
      <td>0.9531</td>
      <td>52.6</td>
      <td>48.1</td>
      <td>15.4</td>
      <td>18.2</td>
      <td>40.7</td>
      <td>30.0</td>
      <td>32.3</td>
      <td>30.4</td>
      <td>53.9</td>
      <td>44.6</td>
      <td>32.7</td>
      <td>36.2</td>
      <td>71.7</td>
      <td>8.6</td>
      <td>2016</td>
    </tr>
    <tr>
      <td>1</td>
      <td>40</td>
      <td>36</td>
      <td>129.1</td>
      <td>93.6</td>
      <td>0.9758</td>
      <td>54.8</td>
      <td>47.7</td>
      <td>12.4</td>
      <td>15.8</td>
      <td>32.1</td>
      <td>23.7</td>
      <td>36.2</td>
      <td>22.4</td>
      <td>54.8</td>
      <td>44.7</td>
      <td>36.5</td>
      <td>37.5</td>
      <td>59.3</td>
      <td>11.3</td>
      <td>2015</td>
    </tr>
    <tr>
      <td>2</td>
      <td>40</td>
      <td>33</td>
      <td>114.4</td>
      <td>90.4</td>
      <td>0.9375</td>
      <td>53.9</td>
      <td>47.7</td>
      <td>14.0</td>
      <td>19.5</td>
      <td>25.5</td>
      <td>24.9</td>
      <td>30.7</td>
      <td>30.0</td>
      <td>54.7</td>
      <td>46.8</td>
      <td>35.2</td>
      <td>33.2</td>
      <td>65.9</td>
      <td>6.9</td>
      <td>2018</td>
    </tr>
    <tr>
      <td>3</td>
      <td>38</td>
      <td>31</td>
      <td>115.2</td>
      <td>85.2</td>
      <td>0.9696</td>
      <td>53.5</td>
      <td>43.0</td>
      <td>17.7</td>
      <td>22.8</td>
      <td>27.4</td>
      <td>28.7</td>
      <td>32.9</td>
      <td>36.6</td>
      <td>52.8</td>
      <td>41.9</td>
      <td>36.5</td>
      <td>29.7</td>
      <td>67.5</td>
      <td>7.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <td>4</td>
      <td>39</td>
      <td>37</td>
      <td>117.8</td>
      <td>86.3</td>
      <td>0.9728</td>
      <td>56.6</td>
      <td>41.1</td>
      <td>16.2</td>
      <td>17.1</td>
      <td>30.0</td>
      <td>26.2</td>
      <td>39.0</td>
      <td>26.9</td>
      <td>56.3</td>
      <td>40.0</td>
      <td>38.2</td>
      <td>29.0</td>
      <td>71.5</td>
      <td>7.7</td>
      <td>2017</td>
    </tr>
    <tr>
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
      <td>1752</td>
      <td>35</td>
      <td>22</td>
      <td>111.2</td>
      <td>94.7</td>
      <td>0.8640</td>
      <td>51.4</td>
      <td>46.9</td>
      <td>19.2</td>
      <td>15.3</td>
      <td>33.9</td>
      <td>27.3</td>
      <td>32.0</td>
      <td>27.6</td>
      <td>52.5</td>
      <td>45.7</td>
      <td>32.9</td>
      <td>32.6</td>
      <td>70.3</td>
      <td>1.9</td>
      <td>2018</td>
    </tr>
    <tr>
      <td>1753</td>
      <td>35</td>
      <td>28</td>
      <td>117.9</td>
      <td>96.6</td>
      <td>0.9081</td>
      <td>51.2</td>
      <td>49.9</td>
      <td>17.9</td>
      <td>20.1</td>
      <td>36.7</td>
      <td>30.8</td>
      <td>37.1</td>
      <td>33.1</td>
      <td>52.9</td>
      <td>49.4</td>
      <td>31.9</td>
      <td>33.7</td>
      <td>71.2</td>
      <td>7.3</td>
      <td>2019</td>
    </tr>
    <tr>
      <td>1754</td>
      <td>36</td>
      <td>31</td>
      <td>122.8</td>
      <td>95.2</td>
      <td>0.9488</td>
      <td>55.3</td>
      <td>48.1</td>
      <td>15.8</td>
      <td>18.0</td>
      <td>31.6</td>
      <td>30.2</td>
      <td>33.3</td>
      <td>34.9</td>
      <td>55.4</td>
      <td>44.7</td>
      <td>36.7</td>
      <td>35.4</td>
      <td>68.8</td>
      <td>9.9</td>
      <td>2019</td>
    </tr>
    <tr>
      <td>1755</td>
      <td>35</td>
      <td>27</td>
      <td>117.4</td>
      <td>94.5</td>
      <td>0.9238</td>
      <td>55.2</td>
      <td>44.8</td>
      <td>17.1</td>
      <td>15.1</td>
      <td>32.1</td>
      <td>26.0</td>
      <td>34.4</td>
      <td>28.1</td>
      <td>54.3</td>
      <td>44.4</td>
      <td>37.8</td>
      <td>30.3</td>
      <td>68.2</td>
      <td>2.1</td>
      <td>2016</td>
    </tr>
    <tr>
      <td>1756</td>
      <td>37</td>
      <td>32</td>
      <td>117.2</td>
      <td>94.9</td>
      <td>0.9192</td>
      <td>57.0</td>
      <td>47.1</td>
      <td>16.1</td>
      <td>17.4</td>
      <td>33.0</td>
      <td>23.1</td>
      <td>32.1</td>
      <td>29.1</td>
      <td>58.2</td>
      <td>44.1</td>
      <td>36.8</td>
      <td>35.0</td>
      <td>70.5</td>
      <td>4.9</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
<p>1757 rows × 20 columns</p>
</div>



#### standardizing the remaining columns, the predictors. First, let's check for variance:


```python
df.var().round(3)
```




    G           6.775
    W          42.844
    ADJOE      53.363
    ADJDE      41.896
    BARTHAG     0.065
    EFG_O       9.800
    EFG_D       8.177
    TOR         3.967
    TORD        4.448
    ORB        16.825
    DRB         9.375
    FTR        23.859
    FTRD       34.821
    2P_O       11.711
    2P_D       10.813
    3P_O        7.520
    3P_D        5.616
    ADJ_T      10.621
    WAB        48.842
    YEAR        2.003
    dtype: float64




```python
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
```


```python
df.var().round(3)
```




    0     0.026
    1     0.030
    2     0.019
    3     0.026
    4     0.068
    5     0.024
    6     0.021
    7     0.021
    8     0.014
    9     0.023
    10    0.019
    11    0.028
    12    0.026
    13    0.019
    14    0.020
    15    0.021
    16    0.022
    17    0.015
    18    0.033
    19    0.125
    dtype: float64




```python
df = df.assign(Y=Y.values)
```


```python
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0000</td>
      <td>0.868421</td>
      <td>0.889313</td>
      <td>0.2725</td>
      <td>0.968152</td>
      <td>0.647059</td>
      <td>0.427136</td>
      <td>0.218978</td>
      <td>0.449438</td>
      <td>0.948339</td>
      <td>...</td>
      <td>0.363946</td>
      <td>0.234332</td>
      <td>0.650602</td>
      <td>0.293617</td>
      <td>0.396825</td>
      <td>0.56875</td>
      <td>0.553435</td>
      <td>0.882507</td>
      <td>0.25</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.0000</td>
      <td>0.947368</td>
      <td>1.000000</td>
      <td>0.2400</td>
      <td>0.991398</td>
      <td>0.754902</td>
      <td>0.407035</td>
      <td>0.000000</td>
      <td>0.314607</td>
      <td>0.630996</td>
      <td>...</td>
      <td>0.496599</td>
      <td>0.016349</td>
      <td>0.686747</td>
      <td>0.297872</td>
      <td>0.597884</td>
      <td>0.65000</td>
      <td>0.080153</td>
      <td>0.953003</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0000</td>
      <td>0.868421</td>
      <td>0.719466</td>
      <td>0.1600</td>
      <td>0.952176</td>
      <td>0.710784</td>
      <td>0.407035</td>
      <td>0.116788</td>
      <td>0.522472</td>
      <td>0.387454</td>
      <td>...</td>
      <td>0.309524</td>
      <td>0.223433</td>
      <td>0.682731</td>
      <td>0.387234</td>
      <td>0.529101</td>
      <td>0.38125</td>
      <td>0.332061</td>
      <td>0.838120</td>
      <td>0.75</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.8750</td>
      <td>0.815789</td>
      <td>0.734733</td>
      <td>0.0300</td>
      <td>0.985049</td>
      <td>0.691176</td>
      <td>0.170854</td>
      <td>0.386861</td>
      <td>0.707865</td>
      <td>0.457565</td>
      <td>...</td>
      <td>0.384354</td>
      <td>0.403270</td>
      <td>0.606426</td>
      <td>0.178723</td>
      <td>0.597884</td>
      <td>0.16250</td>
      <td>0.393130</td>
      <td>0.840731</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.9375</td>
      <td>0.973684</td>
      <td>0.784351</td>
      <td>0.0575</td>
      <td>0.988326</td>
      <td>0.843137</td>
      <td>0.075377</td>
      <td>0.277372</td>
      <td>0.387640</td>
      <td>0.553506</td>
      <td>...</td>
      <td>0.591837</td>
      <td>0.138965</td>
      <td>0.746988</td>
      <td>0.097872</td>
      <td>0.687831</td>
      <td>0.11875</td>
      <td>0.545802</td>
      <td>0.859008</td>
      <td>0.50</td>
      <td>1.0</td>
    </tr>
    <tr>
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
      <td>1752</td>
      <td>0.6875</td>
      <td>0.578947</td>
      <td>0.658397</td>
      <td>0.2675</td>
      <td>0.876907</td>
      <td>0.588235</td>
      <td>0.366834</td>
      <td>0.496350</td>
      <td>0.286517</td>
      <td>0.697417</td>
      <td>...</td>
      <td>0.353741</td>
      <td>0.158038</td>
      <td>0.594378</td>
      <td>0.340426</td>
      <td>0.407407</td>
      <td>0.34375</td>
      <td>0.500000</td>
      <td>0.707572</td>
      <td>0.75</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1753</td>
      <td>0.6875</td>
      <td>0.736842</td>
      <td>0.786260</td>
      <td>0.3150</td>
      <td>0.922069</td>
      <td>0.578431</td>
      <td>0.517588</td>
      <td>0.401460</td>
      <td>0.556180</td>
      <td>0.800738</td>
      <td>...</td>
      <td>0.527211</td>
      <td>0.307902</td>
      <td>0.610442</td>
      <td>0.497872</td>
      <td>0.354497</td>
      <td>0.41250</td>
      <td>0.534351</td>
      <td>0.848564</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1754</td>
      <td>0.7500</td>
      <td>0.815789</td>
      <td>0.879771</td>
      <td>0.2800</td>
      <td>0.963748</td>
      <td>0.779412</td>
      <td>0.427136</td>
      <td>0.248175</td>
      <td>0.438202</td>
      <td>0.612546</td>
      <td>...</td>
      <td>0.397959</td>
      <td>0.356948</td>
      <td>0.710843</td>
      <td>0.297872</td>
      <td>0.608466</td>
      <td>0.51875</td>
      <td>0.442748</td>
      <td>0.916449</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1755</td>
      <td>0.6875</td>
      <td>0.710526</td>
      <td>0.776718</td>
      <td>0.2625</td>
      <td>0.938146</td>
      <td>0.774510</td>
      <td>0.261307</td>
      <td>0.343066</td>
      <td>0.275281</td>
      <td>0.630996</td>
      <td>...</td>
      <td>0.435374</td>
      <td>0.171662</td>
      <td>0.666667</td>
      <td>0.285106</td>
      <td>0.666667</td>
      <td>0.20000</td>
      <td>0.419847</td>
      <td>0.712794</td>
      <td>0.25</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1756</td>
      <td>0.8125</td>
      <td>0.842105</td>
      <td>0.772901</td>
      <td>0.2725</td>
      <td>0.933436</td>
      <td>0.862745</td>
      <td>0.376884</td>
      <td>0.270073</td>
      <td>0.404494</td>
      <td>0.664207</td>
      <td>...</td>
      <td>0.357143</td>
      <td>0.198910</td>
      <td>0.823293</td>
      <td>0.272340</td>
      <td>0.613757</td>
      <td>0.49375</td>
      <td>0.507634</td>
      <td>0.785901</td>
      <td>0.75</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1757 rows × 21 columns</p>
</div>



 

### 40-Modeling

#### Split the dataset into train and test sets

##### it is important to remember that the number of teams that make it to the tournament and those that don't is imbalanced. I will account for this imbalance during the splitting process through the "stratify" method:


```python
# Split DataFrame into
# X_train, X_test, y_train and y_test datasets,
# stratifying on the `target` column
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='Y'),
    df.Y,
    test_size=0.25,
    random_state=42,
    stratify=df.Y
)
```


```python
X_train.head(2)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1037</td>
      <td>0.6875</td>
      <td>0.5</td>
      <td>0.589695</td>
      <td>0.3625</td>
      <td>0.743881</td>
      <td>0.612745</td>
      <td>0.507538</td>
      <td>0.598540</td>
      <td>0.320225</td>
      <td>0.656827</td>
      <td>0.390909</td>
      <td>0.680272</td>
      <td>0.201635</td>
      <td>0.506024</td>
      <td>0.434043</td>
      <td>0.619048</td>
      <td>0.5000</td>
      <td>0.561069</td>
      <td>0.545692</td>
      <td>0.75</td>
    </tr>
    <tr>
      <td>613</td>
      <td>0.5625</td>
      <td>0.5</td>
      <td>0.553435</td>
      <td>0.5550</td>
      <td>0.490425</td>
      <td>0.617647</td>
      <td>0.562814</td>
      <td>0.416058</td>
      <td>0.376404</td>
      <td>0.479705</td>
      <td>0.463636</td>
      <td>0.472789</td>
      <td>0.163488</td>
      <td>0.550201</td>
      <td>0.459574</td>
      <td>0.529101</td>
      <td>0.6375</td>
      <td>0.564885</td>
      <td>0.449086</td>
      <td>0.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test.head(2)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>997</td>
      <td>0.375</td>
      <td>0.236842</td>
      <td>0.421756</td>
      <td>0.6825</td>
      <td>0.199898</td>
      <td>0.411765</td>
      <td>0.728643</td>
      <td>0.510949</td>
      <td>0.410112</td>
      <td>0.527675</td>
      <td>0.5</td>
      <td>0.387755</td>
      <td>0.656676</td>
      <td>0.381526</td>
      <td>0.672340</td>
      <td>0.380952</td>
      <td>0.60625</td>
      <td>0.507634</td>
      <td>0.258486</td>
      <td>0.75</td>
    </tr>
    <tr>
      <td>596</td>
      <td>0.375</td>
      <td>0.236842</td>
      <td>0.475191</td>
      <td>0.7325</td>
      <td>0.218945</td>
      <td>0.470588</td>
      <td>0.703518</td>
      <td>0.233577</td>
      <td>0.308989</td>
      <td>0.284133</td>
      <td>0.7</td>
      <td>0.278912</td>
      <td>0.291553</td>
      <td>0.313253</td>
      <td>0.714894</td>
      <td>0.624339</td>
      <td>0.46875</td>
      <td>0.610687</td>
      <td>0.292428</td>
      <td>0.25</td>
    </tr>
  </tbody>
</table>
</div>



#### I will be using a simple Logistic Regression model for this project, at least for now. My reason is that logistic regression is suitable for when a Y variable takes on only two values. Such a variable is referred to a “binary” or “dichotomous.” “Dichotomous” basically means two categories such as yes/no, defective/non-defective, success/failure, and so on. For this project, my goal is to predict whether a college basketball team will make it to the March Madness tournament (1) or not (0)


```python
# Instantiate LogisticRegression
logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)

# Train the model
logreg.fit(X_train, y_train)

logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')
```

    
    AUC score: 0.9444


##### This is actually impressive and it makes me wonder if I missed anything. The fact that I didn't do much to arrive at this level of accuracy makes me wonder why "Bracketology" is such an unconquered beast in the sports world. But I guess it gets more complicated when people try to predict every single game's outcome. I plan on trying a few other models soon. 

##### Link to Github repo: <a href="https://github.com/alfredprah/college-basketball" class="button button--large">Github: Bracketology</a>
