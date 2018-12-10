from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import sys

def predict(data, data_point):
	

def main(training_set):
	data = pd.read_csv(training_set)
	predict(data, data_point)

if __name__ == '__main__':
	main(*sys.argv[1:])