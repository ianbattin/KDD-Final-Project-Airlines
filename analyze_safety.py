import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def read_safety(path):
    df = pd.read_csv(path)
    df["incidents_85_99"] = df["incidents_85_99"] / df["avail_seat_km_per_week"] * 10000000
    df["fatal_accidents_85_99"] = df["fatal_accidents_85_99"] / df["avail_seat_km_per_week"] * 10000000
    df["fatalities_85_99"] = df["fatalities_85_99"] / df["avail_seat_km_per_week"] * 10000000
    df["incidents_00_14"] = df["incidents_00_14"] / df["avail_seat_km_per_week"] * 10000000
    df["fatal_accidents_00_14"] = df["fatal_accidents_00_14"] / df["avail_seat_km_per_week"] * 10000000
    df["fatalities_00_14"] = df["fatalities_00_14"] / df["avail_seat_km_per_week"] * 10000000
    df2 = pd.DataFrame()
    df2["airline"] = df["airline"]
    df2["incidents"] = df["incidents_85_99"] + df["incidents_00_14"]
    df2["fatalities"] = df["fatalities_85_99"] + df["fatalities_00_14"]
    df2["incidents"] = df2["incidents"] / df2["incidents"].max()
    df2["fatalities"] = df2["fatalities"] / df2["fatalities"].max()
    return df2

def cluster_airlines(df, plot):
    df = read_safety("./airline-safety.csv")
    X = df[["incidents", "fatalities"]]
    y_pred = KMeans(n_clusters=4).fit_predict(X)
    if plot:
        plt.title("Normalized Airline Safety 1984 - 2014")
        plt.xlabel("incidents")
        plt.ylabel("fatalities")
        plt.scatter(df["incidents"], df["fatalities"], c=y_pred, alpha=.5)
        plt.show()
    output = [(df["airline"][i], n) for i, n in enumerate(y_pred)]
    return output

def main():
    df = read_safety("./airline-safety.csv")
    print(cluster_airlines(df, True))

main()
