import pandas as pd
import matplotlib.pyplot as plt
import sys

def graph_airlines_vs_delays(file_path):
	delays = pd.read_csv(file_path)
	counts = delays.groupby('UniqueCarrier').size()

	columns = [
		'UniqueCarrier',
		'CarrierDelay',
		'WeatherDelay',
		'NASDelay',
		'LateAircraftDelay'
	]
	delayCounts = delays[columns].groupby(columns[0]).sum()
	delayCounts = delayCounts.assign(NumDelays = counts)
	delayCounts = delayCounts.sort_values('NumDelays')
	delayCounts['AverageDelay'] = delayCounts[columns[1:]].sum(axis=1) / delayCounts['NumDelays'] * 75000

	plt.style.use('ggplot')
	print(delayCounts)
	delayCounts[columns[1:] + ['AverageDelay']].plot(kind = 'bar')
	plt.xlabel('Airline IATA Codes (sorted by number of delays)')
	plt.ylabel('Total Minutes')
	plt.title('Airline Delays in 2008')
	plt.show()

def main(file_path):
	graph_airlines_vs_delays(file_path)

if __name__ == '__main__':
	main(*sys.argv[1:])