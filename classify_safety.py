import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut

from analyze_fleets import read_fleets
from analyze_safety import cluster_airlines

def main():
    fleets = read_fleets(sys.argv[1])
    safety = cluster_airlines(sys.argv[2], False)
    df = fleets.merge(safety, left_on="Parent Airline", right_on="airline")
    clf = RandomForestClassifier(n_estimators=100)
    loo = LeaveOneOut()
    x = df[["Total Cost Norm", "Average Age Norm"]]
    y = df["label"]
    for train, test in loo.split(x):
        clf.fit(x.loc[train], y.loc[train])
        pred = clf.predict(x.loc[test])
        print("Pred: ", pred)
        print("Actual: ", y.loc[test])

if __name__ == "__main__":
    main()
