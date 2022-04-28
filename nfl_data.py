import numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
  

# open csv file and read it 

import pandas as pd
scores = pd.read_csv('spreadspoke_scores.csv')

#scores = pd.read_csv("../input/nfl-scores-and-betting-data/spreadspoke_scores.csv")

stadiums = pd.read_csv('nfl_stadiums.csv')
#stadiums = pd.read_csv("../input/nfl-scores-and-betting-data/nfl_stadiums.csv")

team_stadiums = pd.read_csv('nfl_team_stadiums.csv')

nfl = scores.join(stadiums.set_index('stadium_name'), on='stadium')
nfl_combined = nfl.join(team_stadiums.set_index('visitor_team'), on='team_away')


## Explore data set
#nfl_combined.plot()
#plt.show()
#nfl_combined.plot(kind = 'scatter', x = 'weather_wind_mph', y = 'score_home')

#unique_team = nfl_combined.team_home.unique()
#unique_stadium = nfl_combined.stadium_location.unique()


## Get games past 2010:
nfl_combined = nfl_combined[nfl_combined.schedule_season > 2010]


    
    
## Add real point differential variable from the perspective of the home team   
point_diff = nfl_combined["score_home"] - nfl_combined["score_away"]
nfl_combined["point_diff"] = point_diff  

## Add outcome variable from the perspectiveof the home team 
nfl_combined['home_outcome'] = np.where(nfl_combined['point_diff'] > 0, 'win', 'loss')
        
## Add predicted point diff, based on the spread from the perspective of the home team
nfl_combined['predicted_point_diff'] = np.where(nfl_combined['team_favorite_id'].astype(str).str[0] == nfl_combined['team_home'].astype(str).str[0],
                                                -(nfl_combined['spread_favorite']), nfl_combined['spread_favorite'])

## Add altitude advantage of the home team, based on altidue difference of away team (doesn't factor in neutral sites)
nfl_combined['altitude_advantage'] = nfl_combined['ELEVATION'] - nfl_combined['visitor_elevation']


##Add geographical advantage of the home team, based on travel distance of away team (doesn't factor in neutral sites)
travel_advantage = []
for ind in nfl_combined.index:
    game_stadium = (nfl_combined['LATITUDE'][ind],nfl_combined['LONGITUDE'][ind])
    visitor_stadium = (nfl_combined['visitor_latitude'][ind],nfl_combined['visitor_longitude'][ind]) 
    
    try:
        t = geodesic(game_stadium,visitor_stadium).mi
        travel_advantage.append(t)
    except ValueError:
        t = None
        travel_advantage.append(t)
    

nfl_combined['travel_advantage'] = travel_advantage 

# Simplify the data frame
nfl_final = nfl_combined[['schedule_season','schedule_week','schedule_playoff','team_home','team_away','altitude_advantage',
                         'travel_advantage','stadium','weather_wind_mph','weather_humidity','predicted_point_diff','point_diff','home_outcome']]

# Drop NA values for analysis
nfl_final = nfl_final.dropna()

'''
my_list = nfl_combined.columns.values.tolist()
for i in my_list:
    print(i)

nfl_final.head()
'''

# Divide up features (x) and classes (y)

x=nfl_final[['altitude_advantage',  'travel_advantage','weather_wind_mph', 'weather_humidity', 'predicted_point_diff' ]]  # Features
y=nfl_final['home_outcome']  

# Split dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(len(x_train))
print(len(x_test))

model=RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=clf.predict(X_test)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_imp = pd.Series(model.feature_importances_,index=x.columns.values).sort_values(ascending=False)
feature_imp


#Source: datacamp random forest "Finding Important Features in Scikit-learn"

%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
