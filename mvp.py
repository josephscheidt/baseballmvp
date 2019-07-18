import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Load in tables needed from baseball database
batting = pd.read_csv("baseballdatabank-2019.2/core/Batting.csv")
fielding = pd.read_csv("baseballdatabank-2019.2/core/Fielding.csv")
pitching = pd.read_csv("baseballdatabank-2019.2/core/Pitching.csv")
teams = pd.read_csv("baseballdatabank-2019.2/core/Teams.csv")
award = pd.read_csv("baseballdatabank-2019.2/core/AwardsSharePlayers.csv")

#merge relevant statistics from batting and pitching tables
mvp = pd.merge(batting[['playerID', 'yearID', 'teamID', 'lgID', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI',
                        'SB', 'BB', 'SO']], 
               pitching[['playerID', 'yearID', 'teamID', 'lgID', 'W', 'L', 'G', 'GS', 'CG', 'SHO', 'SV', 
                    'IPouts', 'H', 'HR', 'BB', 'SO', 'ERA']],
              on = ['playerID', 'yearID', 'teamID', 'lgID'],
              how = 'outer')

#Limit data to modern era of baseball
mvp = mvp[mvp['yearID'] > 1960]

#Get most commonly played position for each player in each season
max_position = fielding.sort_values('POS', ascending=False).drop_duplicates(['playerID','yearID'])

#Add fielding statistics to data frame
mvp = mvp.merge(max_position[['playerID', 'yearID', 'G', 'POS', 'ZR', 'E', 'PO']], how = "left", on = ['playerID', 'yearID'])

#Calculate teams winning percentage, create boolean indicating postseason appearance, add to data frame
teams["win_pct"] = teams["W"]/teams["G"]
team_wins = teams[["yearID", "teamID", "win_pct", "DivWin", "WCWin", "LgWin"]]
post = pd.concat([pd.get_dummies(teams['DivWin'], prefix = "Div"),
                  pd.get_dummies(teams['WCWin'], prefix = "WC"),
                  pd.get_dummies(teams['LgWin'], prefix = "Lg")],
                 axis = 1).drop(['Div_N', 'WC_N', 'Lg_N'], axis = 1)
team_wins = team_wins.assign(post = post.any(axis = 1)).drop(["DivWin", "WCWin", "LgWin"], axis = 1)

mvp = mvp.merge(team_wins, how = 'left', on = ['teamID', 'yearID'])


#Calculate batting average
mvp['BA'] = mvp['H_x'] / mvp['AB']

#Add booleans for round number milestones given outsize importance
mvp['BA_300'] = mvp['BA'] > 0.3
mvp['HR_40'] = mvp['HR_x'] > 39
mvp['RBI_100'] = mvp['RBI'] > 99
mvp['W_20'] = mvp['W'] > 19

#Add boolean for playing for the Colorado Rockies, whose home park has much higher scoring
mvp['Coors'] = mvp['teamID'] == "COL"

#Calculate OPS and WHIP
mvp['OPS'] = (mvp['H_x'] + mvp['BB_x']) / (mvp['AB'] + mvp['BB_x']) + (mvp['H_x'] + 2*mvp['2B'] + 3*mvp['3B'] + 4*mvp['HR_x'])/(mvp['AB'])
mvp['WHIP'] = (mvp['BB_y'] + mvp['H_y']) / mvp['IPouts']

#Replace infinite ERA and WHIP with NA
mvp.replace([np.inf, -np.inf], np.nan, inplace = True)


#Add dependent variable, MVP vote share percentage, to data frame
mvp_award = award[award["awardID"] == "MVP"]
mvp_award = mvp_award.assign(mvp_pct = mvp_award['pointsWon'] / mvp_award['pointsMax'])
mvp = mvp.merge(mvp_award[['yearID', 'lgID', 'playerID', 'mvp_pct']], how = 'left', on = ['yearID', 'lgID', 'playerID'])

#Replace NA values in data frame
mvp['POS'].fillna("H", inplace = True)
mvp['ERA'].fillna(100, inplace = True)
mvp['WHIP'].fillna(100, inplace = True)
mvp.fillna(0, inplace = True)

#Add yearly league ranks for OPS and triple crown categories
mvp['OPS_rank'] = mvp.groupby(['yearID', 'lgID'])['OPS'].rank('dense', ascending = False)
mvp['BA_rank'] = mvp.groupby(['yearID', 'lgID'])['BA'].rank('dense', ascending = False)
mvp['HR_rank'] = mvp.groupby(['yearID', 'lgID'])['HR_x'].rank('dense', ascending = False)
mvp['RBI_rank'] = mvp.groupby(['yearID', 'lgID'])['RBI'].rank('dense', ascending = False)

#drop POS and POS_H for one hot encoding
mvp = pd.concat([mvp, pd.get_dummies(mvp["POS"], prefix = 'POS')], axis = 1).drop(['POS', 'POS_H'], axis = 1)

#Create validation set using the years 2017 and 2018
validation_set = mvp[mvp['yearID'] > 2016]
X_val = validation_set.drop(["playerID", "teamID", "lgID", "mvp_pct"], axis = 1)
y_val = validation_set["mvp_pct"]

#Use remaining data to build model
build_set = mvp[mvp['yearID'] <= 2016]

#Set seed for reproducibility of results
np.random.seed(213436)

#Drop id variables from X matrix, set dependent y vector
X = build_set.drop(["playerID", "teamID", "lgID", "mvp_pct"], axis = 1)
y = build_set["mvp_pct"]

#Split 75/25 for training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Scale data based on training data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

#Create neural network, cross validate to tune layers and size
nn = MLPRegressor(65, activation = 'logistic')

scores = cross_val_score(nn, X_train, y_train, cv = 5)

print(scores)
print(scores.mean())

#Train final neural network, predict test data, print statistics
nn = MLPRegressor(65, activation = 'logistic')

nn.fit(X_train, y_train)
predictions = nn.predict(X_test)

print("MSE:")
print(mean_squared_error(y_test, predictions))

print("R squared:")
print(r2_score(y_test, predictions))

#predict validation set of 2017 and 2018 seasons and get predictions for top 10 finishers in MVP voting
validation_set = validation_set.assign(predicted_pct = nn.predict(X_val))

print('2017')
print(validation_set[(validation_set["yearID"] == 2017) & \
(validation_set['lgID'] == 'AL') ].sort_values('predicted_pct', ascending=False).head(10)['playerID'])
print(validation_set[(validation_set["yearID"] == 2017) & \
(validation_set['lgID'] == 'NL') ].sort_values('predicted_pct', ascending=False).head(10)['playerID'])
print('2018')
print(validation_set[(validation_set["yearID"] == 2018) & \
(validation_set['lgID'] == 'AL') ].sort_values('predicted_pct', ascending=False).head(10)['playerID'])
print(validation_set[(validation_set["yearID"] == 2018) & \
(validation_set['lgID'] == 'NL') ].sort_values('predicted_pct', ascending=False).head(10)['playerID'])
