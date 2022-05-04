#foobar
import streamlit as st
import pandas as pd
import time
import pandas as pd

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
    year = st.radio("Choose a start year",[2015,2016,2017,2018,2019,2020,2021])
    week = st.slider("Choose a start week of season",1,12)

elif choice == "This week":
    year = 2021 
    week = 12

     
scores = pd.read_csv('spreadspoke_scores.csv')
stadiums = pd.read_csv('nfl_stadiums.csv', encoding='latin1')
team_stadiums = pd.read_csv('nfl_team_stadiums.csv')

nfl = scores.join(stadiums.set_index('stadium_name'), on='stadium')
nfl_combined = nfl.join(team_stadiums.set_index('visitor_team'), on='team_away')
     
gather_data = nfl_combined[nfl_combined.schedule_season >= year] 
##gather_data = nfl_combined[nfl_combined.schedule_week >= week] 
     
# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
row2_1, row2_2 = st.columns((2, 1))


with row2_1:
    st.write(
        f"""**All NFL Games from {year}**"""
    )
    st.table(gather_data)

with row2_2:
    st.write("**Gather Insights**")
    clicked = st.button("Get best bets")
    if clicked:
     st.write('Getting bets!')



with st.spinner('Wait for it...'):
    output = 'hello world!'
    
st.success('Done!')
st.balloons()
