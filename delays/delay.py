import pandas as pd
import matplotlib.pyplot as plt
import sys

def graph_airlines_vs_delays(file_path):
	delays = pd.read_csv(file_path)
	counts = delays.groupby('UniqueCarrier').size()

	plt.style.use('ggplot')
	counts.plot(kind = 'bar')
	plt.xlabel('Airline IATA Code')
	plt.ylabel('Number of Delays')
	plt.title('Airline Delays in 2008')
	plt.show()

def main(file_path):
	graph_airlines_vs_delays(file_path)

if __name__ == '__main__':
	main(*sys.argv[1:])