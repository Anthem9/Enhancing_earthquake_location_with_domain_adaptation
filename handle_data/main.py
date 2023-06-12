import pandas as pd
import sys
# import pickle


def catalog2df(filename):
    with open(filename, "r") as f:
        data = f.readlines()
        events = {}
        for d in data:
            event = {}
            ds = d.split(" ")
            # print(ds)

            event['OT'] = ":".join(ds[2:8])
            event['Lat'] = ds[9]
            event['long'] = ds[11]
            event['Depth'] = ds[13].strip()
            # print(event)
            events[ds[0][13:19]] = event
        # print(events)
            # print(len(events))
            # events_dict = {}
        df = pd.DataFrame(events).T
        print(df)
        picklename = filename[:-3] + "pickle"
        df.to_pickle(picklename)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    catalog2df('../phase1/catalog.reference.txt')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
