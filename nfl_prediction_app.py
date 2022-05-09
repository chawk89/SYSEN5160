#foobar
import streamlit as st
import pandas as pd
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np

st.set_page_config(
     page_title="NFL Game Predictor App",
     page_icon="🧊",
     layout="wide",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )



# LAYOUT FOR THE TOP SECTION OF THE APP
row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    st.title("NFL Game Predictor App")
    choice = st.radio("Choose to backtest historical dates or get predictions for this week",["This week","Historical"])

with row1_2:
    st.write(
        """
    ##
    How to use: 
     - Gather information on weekly games.
     - Evaluate the game recommendations based on multiple criteria. 
     - Make a crowd pick to isolate the top recommendation! 
    """
    )


if choice == "Historical":
    year = st.radio("Choose a year",[2015,2016,2017,2018,2019,2020,2021])
    week = st.slider("Choose a week of season",1,12)
    week = str(week)

elif choice == "This week":
    year = 2021 
    week = '4'

     
scores = pd.read_csv('spreadspoke_scores_bets.csv')
stadiums = pd.read_csv('nfl_stadiums.csv', encoding='latin1')
team_stadiums = pd.read_csv('nfl_team_stadiums.csv')

nfl = scores.join(stadiums.set_index('stadium_name'), on='stadium')
nfl_combined = nfl.join(team_stadiums.set_index('visitor_team'), on='team_away')



## Get games past 2010:
nfl_combined = nfl_combined[(nfl_combined.schedule_season > 2010)]


    
    
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

# Simplify the data frames
nfl_final = nfl_combined[['schedule_season','schedule_week','schedule_playoff','team_home','team_away','altitude_advantage',
                         'travel_advantage','stadium','weather_wind_mph','weather_humidity','predicted_point_diff','point_diff','home_outcome',
                         'moneyline_home','moneyline_away','handle_percentage_home','bet_percentage_home']]
                       

# Drop NA values for analysis
#nfl_final = nfl_final.dropna()

# Clean data
column_means = nfl_final.mean()
nfl_final[['weather_wind_mph', 'weather_humidity','altitude_advantage','travel_advantage']] = nfl_final[['weather_wind_mph', 'weather_humidity','altitude_advantage','travel_advantage']].fillna(column_means)


# Divide up features (x) and classes (y)

x=nfl_final[['altitude_advantage',  'travel_advantage','weather_wind_mph', 'weather_humidity', 'predicted_point_diff']]  # Features
y=nfl_final['home_outcome']  

# Split dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(len(x_train))
print(len(x_test))

np.random.seed(123)
model=RandomForestClassifier(n_estimators=120, max_features=2)

# Train the model using the training sets y_pred=clf.predict(X_test)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_imp = pd.Series(model.feature_importances_,index=x.columns.values).sort_values(ascending=False)


#Source: datacamp random forest "Finding Important Features in Scikit-learn"

#%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


current = nfl_final.loc[(nfl_final.schedule_season == year) & (nfl_final.schedule_week == week)]

#>>>>>>> LAYING OUT THE 'GATHER Data' (Top Row) OF THE APP <<<<<<<<<<<
row2_1, row2_2 = st.columns((2, 1))


with row2_1:
    st.write(
        f"""**All NFL Games from {year} & Week: {week}**"""
    )
    st.table(current[['schedule_season','schedule_week','team_home','team_away','predicted_point_diff']])

with row2_2:
    st.write("**Gather Insights**")
    clicked = st.button("Get best bets")
    if clicked:
     st.write('Getting bets!')


#>>> LAYING OUT THE 'Evaluation' ROW OF THE APP <<<<<
row3_1, row3_2 = st.columns((1, 2))

current_predict = current[['altitude_advantage',  'travel_advantage','weather_wind_mph', 'weather_humidity', 'predicted_point_diff']]
predictions = model.predict_proba(current_predict)
current['win_probability'] = [item[1] for item in predictions]
current = current.fillna(value=0)

sns.set_style('whitegrid')

rcParams.update({'font.size': 12})
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']

i = 0
dominated_ids = []

n_samples = len(current)
win_prob = current["win_probability"].tolist()
risk = current["moneyline_home"].tolist()

# Search for dominated design
while i < n_samples:

    #Note: search for a solution in the set that has both a lower weight and a lower delta
    for j,k in zip(win_prob,risk):  
        #print(str(i)+': '+ 'comparing i to j and k')
        if win_prob[i] < j and risk[i] < k: 
            dominated_ids.append(i)
            #print(str(i) + ' is dominated by j and k')
            break
        else:
            continue
           
    i += 1


all_ids = set(list(range(n_samples)))
non_dominated_ids = all_ids.difference(dominated_ids)

index_arr = np.array(list(non_dominated_ids))

mask_arr = np.zeros(n_samples, dtype=int)

mask_arr[index_arr] = 1
current['non-dominated'] = mask_arr

win_prob_dominated = current['win_probability'].loc[current['non-dominated'] == 0]
risk_dominated = current['moneyline_home'].loc[current['non-dominated'] == 0]
win_prob_nondominated = current['win_probability'].loc[current['non-dominated'] == 1]
risk_nondominated = current['moneyline_home'].loc[current['non-dominated'] == 1]

best_bets = current.loc[(current['non-dominated'] == 1) & (current['win_probability'] >= 0.5)]

fig, axs = plt.subplots(2,figsize=(6,9)) 

axs[0].scatter(win_prob_dominated, risk_dominated, c='b')
axs[0].scatter(win_prob_nondominated, risk_nondominated, c='r',marker="o")
#axs[0].set_xlim([0, 15])
#axs[0].set_ylim([0, 5e-3])
axs[0].set_xlabel("Win_prob")
axs[0].axvline(x = 0.5, color = 'b', label = 'Probability Threshold')
axs[0].set_ylabel("Opportunity:Risk Ratio")
axs[0].set_title("Red Teams are Pareto Optimal")

axs[1].bar(feature_imp, feature_imp.index)
axs[1].set_xlabel("Feature Importance Score")
axs[1].set_ylabel("Features")
axs[1].set_title("Top Factors of Win Probability")

plt.tight_layout() 
plt.show()

with row3_1:
    st.write(
        "**Here's the tradespace**"   
    )
    st.pyplot(fig)


with row3_2:
  st.write(
        "**Here are the recommendations**"
    )
  options =  ['0','0','0']
  if clicked:
          st.table(best_bets[['team_home','win_probability','moneyline_home']]) 
          st.session_state.options = st.multiselect(
               'Select the most suitable games: ',
               best_bets.team_home.tolist(),
               best_bets.team_home.tolist() )

user_bets = best_bets[best_bets['team_home'].isin(st.session_state.options)]

#>>>> LAYING OUT THE 'RECOMMNEDATIONS' ROW OF THE APP <<<<<<
row4_1, row4_2 = st.columns((1, 2))


with row4_1:
    st.write(
        "**Post poll on Reddit**"
    )
    st.write(
        "Work in Progress!"
    )


with row4_2:
  st.write(
        "**Get wisdom of the crowd through betting trends**"
    )
  clicked = st.button("Get my final recommendation!")
  pick = user_bets.loc[user_bets['handle_percentage_home'] == user_bets.handle_percentage_home.max()]

  st.dataframe(user_bets[['team_home','bet_percentage_home','handle_percentage_home']])
  if clicked:
   st.dataframe(user_bets[['team_home','bet_percentage_home','handle_percentage_home']])
   st.write(pick['team_home'])
   #st.balloons()

