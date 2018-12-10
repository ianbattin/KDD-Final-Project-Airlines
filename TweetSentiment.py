import pandas as pd
import sys
import operator
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def count_tweet_sentiment(dataset):
  airline_to_sentiment = dict()
  airline_to_score = defaultdict(int)
  airline_to_negative_reason_count = defaultdict(lambda: defaultdict(int))

  for i in range(len(dataset)):
    airline = dataset["airline"][i]
    sentiment = dataset["airline_sentiment"][i]
    reason = dataset["negativereason"][i]

    # Sentiment
    if sentiment == "negative":
      resulting_tuple = (1, 0, 0)
      delta_score = -1
    elif sentiment == "neutral":
      resulting_tuple = (0, 1, 0)
      delta_score = 0
    elif sentiment == "positive":
      resulting_tuple = (0, 0, 1)
      delta_score = 1

    if airline in airline_to_sentiment:
      current = airline_to_sentiment[airline]
      airline_to_sentiment[airline] = tuple(map(operator.add, current, resulting_tuple))
    else:
      airline_to_sentiment[airline] = resulting_tuple

    # Overall Score
    airline_to_score[airline] += delta_score

    # Reason
    airline_to_negative_reason_count[airline][reason] += 1
  
  return airline_to_sentiment, airline_to_score, airline_to_negative_reason_count

def graph_sentiment(data, scores):
  sorted_by_score = sorted([(key, data[key]) for key in data.keys()], key=lambda tup: scores[tup[0]])

  n_groups = len(sorted_by_score)
  negatives = [sentiment[1][0] for sentiment in sorted_by_score]
  neutrals = [sentiment[1][1] for sentiment in sorted_by_score]
  positives = [sentiment[1][2] for sentiment in sorted_by_score]
  
  fix, ax = plt.subplots()
  index = np.arange(n_groups)
  bar_width = 0.25

  rects_neg = plt.bar(index, negatives, bar_width, align='center', color='red', label='Negative')
  rects_neu = plt.bar(index + bar_width, neutrals, bar_width, align='center', color='yellow', label='Neutral')
  rects_pos = plt.bar(index + bar_width * 2, positives, bar_width, align='center', color='green', label='Positive')
  
  plt.xlabel("Airline")
  plt.ylabel("# of Tweets w/ Sentiment")
  plt.title("Tweet Sentiment Count per Airline")
  plt.xticks(index + bar_width, [pair[0] for pair in sorted_by_score])
  plt.legend()
  plt.show()

def graph_reason(data, scores):
  sorted_by_score = sorted([(key, data[key]) for key in data.keys()], key=lambda tup: scores[tup[0]])

  n_groups = len(sorted_by_score)
  
  fix, ax = plt.subplots()
  index = np.arange(n_groups)
  bar_width = 0.07

  badFlight = plt.bar(index, [tup[1]["Bad Flight"] for tup in sorted_by_score], bar_width, align='center', color='red', label="Bad Flight")
  centTell = plt.bar(index + bar_width, [tup[1]["Can't Tell"] for tup in sorted_by_score], bar_width, align='center', color='orange', label="Can't Tell")
  lateFlight = plt.bar(index + bar_width * 2, [tup[1]["Late Flight"] for tup in sorted_by_score], bar_width, align='center', color='yellow', label="Late Flight")
  customerService = plt.bar(index + bar_width * 3, [tup[1]["Customer Service Issue"] for tup in sorted_by_score], bar_width, align='center', color='green', label="Customer Service Issue")
  booking = plt.bar(index + bar_width * 4, [tup[1]["Flight Booking Problems"] for tup in sorted_by_score], bar_width, align='center', color='cyan', label="Flight Booking Problems")
  lostLuggage = plt.bar(index + bar_width * 5, [tup[1]["Lost Luggage"] for tup in sorted_by_score], bar_width, align='center', color='blue', label="Lost Luggage")
  flightAttendant = plt.bar(index + bar_width * 6, [tup[1]["Flight Attendant"] for tup in sorted_by_score], bar_width, align='center', color='purple', label="Flight Attendant")
  cancelled = plt.bar(index + bar_width * 7, [tup[1]["Cancelled Flight"] for tup in sorted_by_score], bar_width, align='center', color='pink', label="Cancelled Flight")
  damaged = plt.bar(index + bar_width * 8, [tup[1]["Damaged Luggage"] for tup in sorted_by_score], bar_width, align='center', color='brown', label="Damaged Luggage")
  cancelled = plt.bar(index + bar_width * 9, [tup[1]["longlines"] for tup in sorted_by_score], bar_width, align='center', color='black', label="Long Lines")

  plt.xlabel("Airline")
  plt.ylabel("# of Tweets w/ Reason")
  plt.title("Negative Reason Count per Airline")
  plt.xticks(index + bar_width * 4, [pair[0] for pair in sorted_by_score])
  plt.legend()
  plt.show()

def classify_tweets(path):
  df = pd.read_csv(path)
  df_reason = pd.get_dummies(df["negativereason"])
  df_sent = pd.get_dummies(df["airline_sentiment"])
  df1 = df_reason.merge(df_sent, left_index=True, right_index=True)
  df1["airline"] = df["airline"]
  clf = RandomForestClassifier(n_estimators=100)
  y = df1["airline"]
  X = df1[["Bad Flight", "Can't Tell", "Cancelled Flight", "Customer Service Issue", "Damaged Luggage", "Flight Attendant Complaints", "Flight Booking Problems", "Late Flight", "Lost Luggage", "longlines", "negative", "neutral", "positive"]]
  print(cross_val_score(clf, X, y, cv=40, scoring='accuracy')) 

def main():
  plt.style.use('ggplot')
  dataset_file = sys.argv[1]
  if sys.argv[2] == "-class":
      classify_tweets(dataset_file)
  dataset = pd.read_csv(dataset_file, usecols=["airline_sentiment", "airline", "negativereason"])
  airline_to_sentiment, airline_to_score, airline_to_reason = count_tweet_sentiment(dataset)

  graph_sentiment(airline_to_sentiment, airline_to_score)
  graph_reason(airline_to_reason, airline_to_score)

if __name__ == "__main__":
    main()
