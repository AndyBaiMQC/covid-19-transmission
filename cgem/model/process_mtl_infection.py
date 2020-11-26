import pandas as pd 
import numpy as np
import pickle

quebec = pd.read_csv("dataset\stable\contact_network\sir_quebec.csv", skiprows=1, names=["date", "total", "active"])
quebec["date"] = pd.to_datetime(quebec.date)

mtl = pd.read_csv("dataset\stable\contact_network\courbe.csv", sep=';', names=["date", "new", "total", "null"])

mtl = mtl.iloc[1:, :-1].dropna()
mtl.new = mtl.new.astype(int)
mtl.total = mtl.total.astype(int)
# print(mtl)
mtl["date"] = pd.to_datetime(mtl.date)

population = 1780000

dates = mtl['date'].unique()
sir = []
# print(sir.shape)

infected = 0
recovered = 0
# cases_to_date = 0

is_hundred = False
hundred_idx = 0
for date_idx, date in enumerate(dates):
    mtl_date = mtl[mtl['date'] == date]
    quebec_date = quebec.loc[quebec["date"] == date]
    if not quebec_date.empty:
        infected = mtl_date["total"].sum()
        
        if not is_hundred:
            if infected > 100:
                is_hundred = True
                hundred_idx = date_idx
                # print(len(dates), hundred_idx)
                sir = np.zeros((len(dates)-hundred_idx, 4))
                # print(sir.shape)
                # exit
            
        else:
            percent_recovered = 1-(quebec_date["active"]/quebec_date["total"])

            recovered = np.round(percent_recovered * infected)
            # print(date_idx)
            sir[date_idx-hundred_idx][0] = population - infected - recovered
            sir[date_idx-hundred_idx][1] = 0
            sir[date_idx-hundred_idx][2] = infected - recovered
            sir[date_idx-hundred_idx][3] = recovered

print(sir)
# print(sir)
# pickle.dump(sir, open("seir_mtl.pkl", "wb"))
