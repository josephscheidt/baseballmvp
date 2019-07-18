# baseballmvp
A model for predicting the top finishers in baseball's MVP voting

## Model Selection
Predicting voting for baseball’s MVP award resists straightforward regression modeling. Complex variable interactions demand a model beyond a simple beta regression model to predict the vote share for each candidate with any degree of accuracy. Given its capacity for universal approximation and ability to efficiently define complex interactions through gradient descent, the neural network makes an ideal choice for this problem.

## Features
Though neural networks are universal approximators, in the interest of improving the efficiency and accuracy of this neural network, care was taken in feature selection. Batting average was calculated, and Boolean indicator variables were added for certain round numbers found to be significant in exploratory analysis. Team success measures and fielding position indicator variables were added. All data were scaled based on a model training set. Original data comes from the Lahman baseball database. The following features were used in the network:
•	Batting: G, AB, R, H, 2B, 3B, HR, RBI, SB, BB, SO, BA, .3BA, 40HR, 100RBI, OPS, triple crown ranks
•	Fielding: POS, ZR, E
•	Pitching: G, W, L, GS, CG, SHO, SV, IPouts, H, HR, BB, SO, ERA, WHIP
•	Team: win percentage, playoff appearance, Coors

## Training
A training set consisting of 75 percent of the data between 1961 and 2016 was used for training, with the remaining 25 percent saved for testing. The years 2017 and 2018 were saved as a validation set to evaluate the utility of the model in making predictions based on a full set of player seasons. The network consisted of a single hidden layer of 65 neurons, which was optimized using 5 fold cross validation on the training set.

## Results
The mean squared error on the test data set was 0.0017, with an R squared of 0.505. To evaluate utility, predictions were made for MVP winners and top 10 finishers for each league in the 2017 and 2018 seasons. The model correctly predicted 2 out of 4 winners (with the other two finishing 4th and 4th), 14 out of the 20 top 5 finishers, and 25 out of the 40 top 10 finishers in each league for each year. 

## Next Steps
Incorporating advanced defensive statistics in the next model iteration may improve results, as the model tended to overestimate the vote proportions of poor defensive players and underestimate the vote proportions of excellent defenders, which may be the result of the increasing awareness of these measures among the voting body in recent years, especially as these statistics are a key component in Wins Above Replacement, a single statistic gaining in popularity for evaluating players. In addition, the vote totals of pitchers seemed to be underestimated, and offers another avenue for investigation and improvement. In all, the correlated nature of these errors bodes well for the model’s potential with further work.
