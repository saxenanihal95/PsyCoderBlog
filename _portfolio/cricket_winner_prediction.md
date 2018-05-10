---
excerpt: "Predictin match winner in Cricket using machine learning"
categories:
  - Machine Learning
tags:
  - machine-learning
  - logistic-regression
  - mlp-classifier
header:
  teaser: assets/images/PsyCoder1.png
    
---

# Predictin match winner in Cricket using machine learning

## Data Preprocessing


```python
import pandas as pd
```


```python
dataset=pd.read_csv('ipl.csv',index_col=0)
```


```python
dataset = dataset.drop(columns=['gender', 'match_type','date','umpire_1','umpire_2','player of the match','win_by_runs','win_by_wickets'])
```


```python
# columns with missing values
dataset.columns[dataset.isnull().any()]
```




    Index(['city'], dtype='object')




```python
# replace missing column with mode value
dataset['city'].fillna(dataset['city'].mode()[0], inplace=True)
```


```python
dataset.columns[dataset.isnull().any()]

dataset.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)
```


```python
dataset.head(5)
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
      <th>city</th>
      <th>team 1</th>
      <th>team 2</th>
      <th>team_1_batting_average</th>
      <th>team_1_bowling_average</th>
      <th>team_2_batting_average</th>
      <th>team_2_bowling_average</th>
      <th>toss_decision</th>
      <th>toss_winner</th>
      <th>venue</th>
      <th>winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bangalore</td>
      <td>KKR</td>
      <td>RCB</td>
      <td>5.0</td>
      <td>5.000000</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>field</td>
      <td>RCB</td>
      <td>M Chinnaswamy Stadium</td>
      <td>KKR</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chandigarh</td>
      <td>CSK</td>
      <td>KXIP</td>
      <td>5.0</td>
      <td>54.000000</td>
      <td>5.0</td>
      <td>47.0</td>
      <td>bat</td>
      <td>CSK</td>
      <td>Punjab Cricket Association Stadium, Mohali</td>
      <td>CSK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Delhi</td>
      <td>RR</td>
      <td>DD</td>
      <td>5.0</td>
      <td>21.000000</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>bat</td>
      <td>RR</td>
      <td>Feroz Shah Kotla</td>
      <td>DD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mumbai</td>
      <td>MI</td>
      <td>RCB</td>
      <td>28.0</td>
      <td>14.750000</td>
      <td>3.0</td>
      <td>19.0</td>
      <td>bat</td>
      <td>MI</td>
      <td>Wankhede Stadium</td>
      <td>RCB</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kolkata</td>
      <td>DC</td>
      <td>KKR</td>
      <td>5.0</td>
      <td>5.666667</td>
      <td>50.0</td>
      <td>10.0</td>
      <td>bat</td>
      <td>DC</td>
      <td>Eden Gardens</td>
      <td>KKR</td>
    </tr>
  </tbody>
</table>
</div>




```python
def createDict(series) :
    
    dictionary={}
    
    i=0
    
    for ser in series :
        if(ser in dictionary) :
            continue
        dictionary[ser]=i
        i=i+1
        
    return dictionary
```


```python
teamDict=createDict(dataset['team 1'])

cityDict=createDict(dataset['city'])

venueDict=createDict(dataset['venue'])

tossDecisionDict=createDict(dataset['toss_decision'])

winnerDict=dict(teamDict)

winnerDict['tie']=14

winnerDict['no result']=15
```


```python
venueDict
```




    {'M Chinnaswamy Stadium': 0,
     'Punjab Cricket Association Stadium, Mohali': 1,
     'Feroz Shah Kotla': 2,
     'Wankhede Stadium': 3,
     'Eden Gardens': 4,
     'Sawai Mansingh Stadium': 5,
     'Rajiv Gandhi International Stadium, Uppal': 6,
     'MA Chidambaram Stadium, Chepauk': 7,
     'Dr DY Patil Sports Academy': 8,
     'Newlands': 9,
     "St George's Park": 10,
     'Kingsmead': 11,
     'SuperSport Park': 12,
     'Buffalo Park': 13,
     'New Wanderers Stadium': 14,
     'De Beers Diamond Oval': 15,
     'OUTsurance Oval': 16,
     'Brabourne Stadium': 17,
     'Sardar Patel Stadium, Motera': 18,
     'Barabati Stadium': 19,
     'Vidarbha Cricket Association Stadium, Jamtha': 20,
     'Himachal Pradesh Cricket Association Stadium': 21,
     'Nehru Stadium': 22,
     'Holkar Cricket Stadium': 23,
     'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 24,
     'Subrata Roy Sahara Stadium': 25,
     'Shaheed Veer Narayan Singh International Stadium': 26,
     'JSCA International Stadium Complex': 27,
     'Sheikh Zayed Stadium': 28,
     'Sharjah Cricket Stadium': 29,
     'Dubai International Cricket Stadium': 30,
     'Maharashtra Cricket Association Stadium': 31,
     'Punjab Cricket Association IS Bindra Stadium, Mohali': 32,
     'Saurashtra Cricket Association Stadium': 33,
     'Green Park': 34}




```python
encode = {
'team 1': teamDict,
'team 2': teamDict,
'toss_winner': teamDict,
'winner': winnerDict,
'city':cityDict,
'venue':venueDict,
'toss_decision': tossDecisionDict    
 }
dataset.replace(encode, inplace=True)
```


```python
def prediction(Model,X_train,y_train,X_test,y_test) :
    
    clf=Model()
    
    clf.fit(X_train,y_train)
    
    print(clf.score(X_test,y_test))
    
    return clf
```


```python
def predictWinner():    
    
    from sklearn.neural_network import MLPClassifier

    from sklearn.svm import LinearSVC

    from sklearn.linear_model import LogisticRegression

    from sklearn.ensemble import RandomForestClassifier

    clf_A = prediction(MLPClassifier,X_train,y_train,X_test,y_test)

    clf_B = prediction(LinearSVC,X_train,y_train,X_test,y_test)

    clf_C = prediction(LogisticRegression,X_train,y_train,X_test,y_test)

    clf_D = prediction(RandomForestClassifier,X_train,y_train,X_test,y_test)
    
```


```python
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression

def buildModel(dataset,team1,team2) :

    
    dataset=dataset[
        ((dataset['team 1']==team1)&(dataset['team 2']==team2) | 
         (dataset['team 1']==team2)&(dataset['team 2']==team1))
    ]


    winner = dataset['winner']

    features = dataset.drop('winner',axis=1)

    features = pd.get_dummies(features)

    clf=LogisticRegression()

    clf.fit(features,winner)

    return clf
```


```python
def getPrediction(city,team1,team2,team1_batting_avg,team1_bowling_avg,team2_batting_avg,team2_bowling_avg,toss_decision,toss_winner,venue) :

    predictionSet = pd.DataFrame({
        'city':cityDict[city],
        'team 1':teamDict[team1],
        'team 2':teamDict[team2],
        'team_1_batting_average':team1_batting_avg,
        'team_1_bowling_average':team1_bowling_avg,
        'team_2_batting_average':team2_batting_avg,
        'team_2_bowling_average':team2_bowling_avg,
        'toss_decision':[toss_decision],
        'toss_winner':teamDict[toss_winner],
        'venue':venueDict[venue]
    })

    predictionSet = pd.get_dummies(predictionSet)
    
    clf=buildModel(dataset,teamDict[team1],teamDict[team2])
    
    prediction=clf.predict(predictionSet)
    
    for key,value in teamDict.items() :
        
        if(value==prediction) :
            
            print(key)
```


```python

getPrediction('Bangalore','KKR','RCB',5.0,5.000000,3.0,12.0,'field','RCB','M Chinnaswamy Stadium')
```

    KKR



```python
getPrediction('Chandigarh','KXIP','CSK',5.0,54.000000,5.0,47.0,'bat','CSK','Punjab Cricket Association Stadium, Mohali')
```

    CSK



```python
getPrediction('Delhi','DD','RR',5.0,21.000000,5.0,5.0,'bat','RR','Feroz Shah Kotla')
```

    DD

