---
title:  "Kaggle #1 -Walmart Recruiting - Store Sales Forecasting"
search: True
categories: 
- kaggle
---

# Kaggle #1 -Walmart Recruiting - Store Sales Forecasting"




## Lean Version


```python
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip")
test = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip")

display(train, test)

train2 = train.drop(['Date','Weekly_Sales'],1)
test2 = test.drop('Date',1)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_jobs=4)

%%time
rf.fit(train2,train['Weekly_Sales'])

result = rf.predict(test2)

sub = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip")
sub["Weekly_Sales"] = result
sub.to_csv("sub_0824.csv",index = 0)

```

## Main Version


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip
    /kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip
    /kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv
    /kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip
    /kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip


Check the files that are provided.

Read data.


```python
train = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip")
test = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip")
```

Visually check the form of data.


```python
train
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
      <th>Store</th>
      <th>Dept</th>
      <th>Date</th>
      <th>Weekly_Sales</th>
      <th>IsHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2010-02-05</td>
      <td>24924.50</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2010-02-12</td>
      <td>46039.49</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2010-02-19</td>
      <td>41595.55</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2010-02-26</td>
      <td>19403.54</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>2010-03-05</td>
      <td>21827.90</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>421565</th>
      <td>45</td>
      <td>98</td>
      <td>2012-09-28</td>
      <td>508.37</td>
      <td>False</td>
    </tr>
    <tr>
      <th>421566</th>
      <td>45</td>
      <td>98</td>
      <td>2012-10-05</td>
      <td>628.10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>421567</th>
      <td>45</td>
      <td>98</td>
      <td>2012-10-12</td>
      <td>1061.02</td>
      <td>False</td>
    </tr>
    <tr>
      <th>421568</th>
      <td>45</td>
      <td>98</td>
      <td>2012-10-19</td>
      <td>760.01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>421569</th>
      <td>45</td>
      <td>98</td>
      <td>2012-10-26</td>
      <td>1076.80</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>421570 rows × 5 columns</p>
</div>




```python
test
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
      <th>Store</th>
      <th>Dept</th>
      <th>Date</th>
      <th>IsHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-02</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-09</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-16</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-23</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-30</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115059</th>
      <td>45</td>
      <td>98</td>
      <td>2013-06-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>115060</th>
      <td>45</td>
      <td>98</td>
      <td>2013-07-05</td>
      <td>False</td>
    </tr>
    <tr>
      <th>115061</th>
      <td>45</td>
      <td>98</td>
      <td>2013-07-12</td>
      <td>False</td>
    </tr>
    <tr>
      <th>115062</th>
      <td>45</td>
      <td>98</td>
      <td>2013-07-19</td>
      <td>False</td>
    </tr>
    <tr>
      <th>115063</th>
      <td>45</td>
      <td>98</td>
      <td>2013-07-26</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>115064 rows × 4 columns</p>
</div>



Check both at the same time.


```python
display(train, test)
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
      <th>Store</th>
      <th>Dept</th>
      <th>Date</th>
      <th>Weekly_Sales</th>
      <th>IsHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2010-02-05</td>
      <td>24924.50</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2010-02-12</td>
      <td>46039.49</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2010-02-19</td>
      <td>41595.55</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2010-02-26</td>
      <td>19403.54</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>2010-03-05</td>
      <td>21827.90</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>421565</th>
      <td>45</td>
      <td>98</td>
      <td>2012-09-28</td>
      <td>508.37</td>
      <td>False</td>
    </tr>
    <tr>
      <th>421566</th>
      <td>45</td>
      <td>98</td>
      <td>2012-10-05</td>
      <td>628.10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>421567</th>
      <td>45</td>
      <td>98</td>
      <td>2012-10-12</td>
      <td>1061.02</td>
      <td>False</td>
    </tr>
    <tr>
      <th>421568</th>
      <td>45</td>
      <td>98</td>
      <td>2012-10-19</td>
      <td>760.01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>421569</th>
      <td>45</td>
      <td>98</td>
      <td>2012-10-26</td>
      <td>1076.80</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>421570 rows × 5 columns</p>
</div>



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
      <th>Store</th>
      <th>Dept</th>
      <th>Date</th>
      <th>IsHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-02</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-09</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-16</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-23</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>2012-11-30</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115059</th>
      <td>45</td>
      <td>98</td>
      <td>2013-06-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>115060</th>
      <td>45</td>
      <td>98</td>
      <td>2013-07-05</td>
      <td>False</td>
    </tr>
    <tr>
      <th>115061</th>
      <td>45</td>
      <td>98</td>
      <td>2013-07-12</td>
      <td>False</td>
    </tr>
    <tr>
      <th>115062</th>
      <td>45</td>
      <td>98</td>
      <td>2013-07-19</td>
      <td>False</td>
    </tr>
    <tr>
      <th>115063</th>
      <td>45</td>
      <td>98</td>
      <td>2013-07-26</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>115064 rows × 4 columns</p>
</div>


Remove the target column and unnecessary columns.<br>



```python
train2 = train.drop(['Date','Weekly_Sales'],1) #,왼쪽은 지울 컬럼, 오른쪽은 행을 지울건지(0)(기본값) 열을 지울건지(1)
```

Make the form the same.


```python
test2 = test.drop('Date',1)
```


```python
train.dtypes
```




    Store             int64
    Dept              int64
    Date             object
    Weekly_Sales    float64
    IsHoliday          bool
    dtype: object




```python
from sklearn.ensemble import RandomForestRegressor
```


```python
rf = RandomForestRegressor(n_jobs=4) #n_jobs=4 : cpu 4개 써라
```


```python
%%time
rf.fit(train2,train['Weekly_Sales'])
```

    CPU times: user 42.4 s, sys: 159 ms, total: 42.5 s
    Wall time: 11.5 s





    RandomForestRegressor(n_jobs=4)




```python
result = rf.predict(test2)
```


```python
result
```




    array([22307.53041113, 22307.53041113, 22307.53041113, ...,
             566.03253747,   566.03253747,   566.03253747])




```python
sub = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip")
```


```python
sub
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
      <th>Id</th>
      <th>Weekly_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1_1_2012-11-02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1_1_2012-11-09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1_1_2012-11-16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1_1_2012-11-23</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1_1_2012-11-30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115059</th>
      <td>45_98_2013-06-28</td>
      <td>0</td>
    </tr>
    <tr>
      <th>115060</th>
      <td>45_98_2013-07-05</td>
      <td>0</td>
    </tr>
    <tr>
      <th>115061</th>
      <td>45_98_2013-07-12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>115062</th>
      <td>45_98_2013-07-19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>115063</th>
      <td>45_98_2013-07-26</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>115064 rows × 2 columns</p>
</div>




```python
sub["Weekly_Sales"] = result
```


```python
sub
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
      <th>Id</th>
      <th>Weekly_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1_1_2012-11-02</td>
      <td>22307.530411</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1_1_2012-11-09</td>
      <td>22307.530411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1_1_2012-11-16</td>
      <td>22307.530411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1_1_2012-11-23</td>
      <td>25279.409294</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1_1_2012-11-30</td>
      <td>22307.530411</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115059</th>
      <td>45_98_2013-06-28</td>
      <td>566.032537</td>
    </tr>
    <tr>
      <th>115060</th>
      <td>45_98_2013-07-05</td>
      <td>566.032537</td>
    </tr>
    <tr>
      <th>115061</th>
      <td>45_98_2013-07-12</td>
      <td>566.032537</td>
    </tr>
    <tr>
      <th>115062</th>
      <td>45_98_2013-07-19</td>
      <td>566.032537</td>
    </tr>
    <tr>
      <th>115063</th>
      <td>45_98_2013-07-26</td>
      <td>566.032537</td>
    </tr>
  </tbody>
</table>
<p>115064 rows × 2 columns</p>
</div>




```python
sub.to_csv("sub_0824.csv",index = 0) #index기본값은 1, 이게 기본값1로 돼있으면 옆에 index컬럼 생김
```
