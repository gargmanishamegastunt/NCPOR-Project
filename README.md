# NCPOR-Summer-2020
Deeplearning models to predict blizzards 
The separate models made from the data provided by NCPOR center Goa, are named as RNN and LSTM model- paper respectively. 

The Blizzard_django folder contains the project that was built to see the number of blizzards occuring in a user inputed month. The prediction model inferences can also be seen via the user via the website built for the user. 
All the data was provided by NCPOR Goa taken from the Indian stations at Bharati (Antarctica)


LSTM MODEL:
The code aims at testing which variable out of Temperature, Air Pressure, Wind speed, wind direction and relative humidity taken independently is the best to predict blizzards. The code cleans each of the given variables one by ones, first by removing outliers by taking the trailing rolling mean, then scales them by using a min-max scaler, the nan values are removed, then the data is normalized. then we split the data into sequences of 7 where one on the variable is the past blizzard occurrence itself. The data is then split into training and test data set and fed into an LSTM model. We use a binary focal loss as we have an imbalanced data set, class weights are assigned also taking this fact into consideration. Next, the model accuracy and model loss are checked. 
