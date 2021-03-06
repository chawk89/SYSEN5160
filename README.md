# SYSEN5160 Final Project: NFL Game Predicton App

Colby Hawker (ch955@cornell.edu) and Kevin Lee (kl938@cornell.edu), 2022.
Cornell University

# Abstract

The purpose of the application is to demonstrate a 2-tiered decision system that introduces humans-in-the-loop to add experiential knowledge to a prediction workflow. The application is meant to predict the team most likely to win in a given week during the NFL season. 

The a posteriori decision-making process works as follows:
- Data was sourced from Kaggle (2022) to develop features for an ensemble decision tree model. The decision tree uses features such as weather and travel fatigue to be used in tandem with Sportsbooks spreads. The prediction and moneyline are used to present tradeoffs of likelihood vs the overall risk/opportunity for a user. Each team is plotted on a tradespace graph, and all non-dominated team (with at least 50% probability of winning) are added to the selection list. The user can adjust this list.
- The second part of the processs envokes the 'wisdom of the crowd'. The game that has the highest volume of bets (via DraftKings Sportsbook) wins the pick. 


# Updates:
Week of May 16th, 2022:
- Accounting for missing data: Estimated moneyline data for all historical years with a linear model. Now the app works for all historical years <= 2020.
- Accounting for missing data: If there are more than one final 'picks', or no betting data available, default to the win probability.
- Changed the final pick to maximize for bet percentage rather than handle percentage.


# Sources:
- https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data
- DraftKings.com
- Some NFL stadium and geographical data was mined from official sources (ie. NFL.com, Google Geo)

# Screenshots

<img width="1187" alt="Screen Shot 2022-05-09 at 7 58 27 PM" src="https://user-images.githubusercontent.com/32115931/167534054-960be7a8-35c3-4936-8c57-7719d0ce71d3.png">

<img width="1566" alt="Screen Shot 2022-05-09 at 7 59 40 PM" src="https://user-images.githubusercontent.com/32115931/167534082-e409792b-b258-41de-b2af-c5d534987265.png">

<img width="1045" alt="Screen Shot 2022-05-09 at 7 59 57 PM" src="https://user-images.githubusercontent.com/32115931/167534088-58bc122d-a8fb-45ea-b86f-07d9e946a323.png">
