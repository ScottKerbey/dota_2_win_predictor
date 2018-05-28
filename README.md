# dota_2_win_predictor
Predicting which team will win based on the heroes picked using a neural network

Dependencies:
Tensorflow, Numpy, MySql.Connector and dota2api

Requires a MySQL server to save match details

Create a text file - api_key.txt - to put your dota 2 API key in.
You can get a key from here: https://steamcommunity.com/dev/apikey

Create a text file - config_file.txt - to put your username and password in for connecting to the database
You can edit the config dictionary in win_classifier.py and setup_database.ipynb to configure you connection

Run setup_database.ipynb in a jupyter notebook to set up database tables and populate them with data

Run win_classifier with --train flag to train the neural network
You can provide other flags to tweak the operation of the training

Run win_classifier with --predict to run a prediction of a match
e.g. >python win_classifier.py --predict="1,2,3,4,5,6,7,8,9,10" 
with the first 5 numbers being the hero_ids of the radiant heroes and the second 5 dire.
The output will be the probability that radiant will win.
