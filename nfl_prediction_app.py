#foobar
import streamlit as st
import pandas as pd
import time



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



# Space out the coluumns so the first one is 2x the size of the other one
c1, c2 = st.columns((2, 1))

header = st.container()
with header:
    choice = st.radio("Choose to backtest historical dates or get predictions for this week",["This week","Historical"])

with c1:
     article = "Anger and confusion overflowed at the Olympic mixed-team ski jumping final in China after five female competitors were disqualified from the event by officials who said their jumpsuits didn't comply with the rules."
     input = st.text_area("Insert Text", article)

#choice = st.sidebar.radio("Choose to backtest historical dates or get predictions for this week",["This week","Historical"])

#st.sidebar.image("https://sportshub.cbsistatic.com/i/r/2021/12/06/e072d88c-0cd9-4390-b919-353d85710ebb/thumbnail/770x433/94d78d1afd5713db52124e1317f4e8cb/beijing-2022.jpg")    
#st.sidebar.video("https://www.youtube.com/watch?v=SPKckEXhWwU")

word_count = len(input.split())

with c2:
     st.write('Character count: ', len(input))
     st.write('Word count: ', word_count)
     "Please allow a few seconds for me to digest! Any radio button selection may add to the time."

     if len(input) > 2000:
          st.write("Input may be just a bit too long!")



gold = pd.DataFrame({ 'Year': ['2006','2010','2014','2018'],
                    'Medals': [9,9,9,9]
                       })

silver = pd.DataFrame({ 'Year':['2006','2010','2014','2018'],
                    'Medals': [9,15,9,8]
                       })

bronze = pd.DataFrame({ 'Year': ['2006','2010','2014','2018'],
                    'Medals': [7,13,10,6]
                       })


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
