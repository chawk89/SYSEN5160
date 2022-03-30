import pandas as pd 
import matplotlib.pyplot as plt
import SALib
from geopy.distance import geodesic
  
# Test loading the lat-long data 
stadium_a = (22.5, 88.3639)
stadium_b = (28.7041, 77.1025)
  
# Print the distance calculated in miles
print(geodesic(stadium_a, stadium_b).mi)

# open csv file and read it 
scores = pd.read_csv("../input/nfl-scores-and-betting-data/spreadspoke_scores.csv")


stadiums = pd.read_csv("../input/nfl-scores-and-betting-data/nfl_stadiums.csv") 
nfl_combined = scores.join(stadiums.set_index('stadium_name'), on='stadium')

nfl_combined.plot()

plt.show()

nfl_combined.plot(kind = 'scatter', x = 'weather_wind_mph', y = 'score_home')
