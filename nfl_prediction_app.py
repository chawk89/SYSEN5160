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

"""
# NFL Game Predictor
How to use: 
- Gather information on weekly games.
- Evaluate the game recommendations based on multiple criteria. 
- Make a crowd pick to isolate the top recommendation! 
"""

scores = pd.read_csv('spreadspoke_scores.csv')
stadiums = pd.read_csv('nfl_stadiums.csv')
team_stadiums = pd.read_csv('nfl_team_stadiums.csv')

nfl = scores.join(stadiums.set_index('stadium_name'), on='stadium')
nfl_combined = nfl.join(team_stadiums.set_index('visitor_team'), on='team_away')



## Get games past 2020:
nfl_combined = nfl_combined[nfl_combined.schedule_season > 2020]




# Space out the coluumns so the first one is 2x the size of the other one
c1, c2 = st.columns((2, 1))

header = st.container()
with header:
    choice = st.radio("Choose to backtest historical dates or get predictions for this week",["This week","Historical"])

with c1:
     st.table(scores)

with c2:
     clicked = st.button("Get best bets")
     if clicked:
          st.write('Getting bets!')





if choice == "This week":
    st.sidebar.dataframe(gold)
    st.sidebar.table(gold)
    st.sidebar.line_chart(gold.rename(columns={'Year':'index'}).set_index('index'))
    st.sidebar.bar_chart(gold.rename(columns={'Year':'index'}).set_index('index'))
elif choice == "Historical":
    st.sidebar.dataframe(silver)
    st.sidebar.table(silver)
    st.sidebar.line_chart(silver.rename(columns={'Year':'index'}).set_index('index'))
    st.sidebar.bar_chart(siver.rename(columns={'Year':'index'}).set_index('index'))
else:
    st.sidebar.dataframe(bronze)
    st.sidebar.table(bronze)
    st.sidebar.line_chart(bronze.rename(columns={'Year':'index'}).set_index('index'))
    st.sidebar.bar_chart(bronze.rename(columns={'Year':'index'}).set_index('index'))


with st.spinner('Wait for it...'):
    output = 'hello world!'
    
st.success('Done!')


    
c = st.empty()
c.header('Summary:')
st.subheader(output)
st.balloons()
