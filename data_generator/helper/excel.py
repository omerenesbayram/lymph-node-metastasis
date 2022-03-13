import numpy as np
import pandas as pd


def get_label(name, type, path):
    df = pd.read_excel(path)
    df["Name"] = df["Name"].str.lower()
    df = df.loc[df['Name'] == name]
    
    l1 = df[["2R"]].iloc[0].to_numpy()
    l2 = df[["4R"]].iloc[0].to_numpy()
    l3 = df[[7]].iloc[0].to_numpy()
    l4 = df[["2L"]].iloc[0].to_numpy()
    l5 = df[["4L"]].iloc[0].to_numpy()

    df = l1 or l2 or l3 or l4 or l5

    if(type == "train"):
        if df == 0:
            return np.array([1,0])
        else:
            return np.array([0,1])
    if(type == "predict"):
        return df


def get_names(path):
    df = pd.read_excel(path)
    df["Name"] = df["Name"].str.lower()

    df = df[["Name", "Uygunluk"]]
    df = df[df.Uygunluk == 1]
    df = df.dropna()

    df = df["Name"]
    df = df.to_numpy()

    pos = []
    neg = []

    for name in df:
        label = get_label(name, "predict", path)
        if(label == 1):
            pos.append(name)
        else:
            neg.append(name)

    return pos, neg
