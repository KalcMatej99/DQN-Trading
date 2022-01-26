import pandas as pd
import json
import os
list_of_files = []
for root, dirs, files in os.walk("./Data"):
	for file in files:
		list_of_files.append(os.path.join(root,file))
for name in list_of_files:
    print(name)
    if ".json" in name:
        with open(name, "r") as read_file:
            emps = json.load(read_file)
            df = pd.DataFrame.from_dict(emps, orient='columns')
            df.columns = ["Date", "Open", "High",       "Low",     "Close",     "Volume"]    
            df["Date"] = pd.to_datetime(df['Date'], unit='ms')
            df["Adj Close"] = df["Close"]
            df = df.set_index("Date")
            os.remove(name)
            df.to_csv(name.replace(".json", ".csv"), index="Date")
