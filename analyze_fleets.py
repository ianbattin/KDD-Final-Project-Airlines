import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

def read_fleets(path):
    df = pd.read_csv(path)
    df1 = df[["Parent Airline", "Current", "Future", "Total", "Total Cost (Current)"]].fillna(0)
    df2 = df[["Parent Airline", "Average Age"]].dropna()
    df1["Total Cost"] = df1["Total Cost (Current)"].str.replace("$", "").str.replace(",", "").fillna("0").astype("int64")
    df1 = df1.groupby("Parent Airline").sum()
    df2 = df2.groupby("Parent Airline").mean()
    df = df1.merge(df2, on="Parent Airline")
    df["Total Cost Norm"] = (df["Total Cost"] - df["Total Cost"].mean()) / (df["Total Cost"].max() - df["Total Cost"].min())
    df["Average Age Norm"] = (df["Average Age"] - df["Average Age"].mean()) / (df["Average Age"].max() - df["Average Age"].min())
    return df[["Total Cost", "Average Age", "Total Cost Norm", "Average Age Norm"]].reset_index()

def plot_fleets(df):
    plt.style.use("ggplot")
    plt.title("Aggregated Fleet Data per Airline")
    plt.xlabel("Average Age of Fleet")
    plt.ylabel("Total Cost of Fleet")
    plt.scatter(df["Average Age"], df["Total Cost"], alpha=.75)
    plt.show()

def main():
    df = read_fleets(sys.argv[1])
    plot_fleets(df)
    print(df[["Parent Airline", "Total Cost", "Average Age"]])

main()
