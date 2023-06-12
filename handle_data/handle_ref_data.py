import re
import pandas as pd

def extract_info_from_file(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            match = re.match(r'phaseworm2019(.{6}).obs OT \d{4} \d{2} \d{2} \d{2} \d{2} \d{2}\.\d{6} Lat (-?\d+\.\d+) Long (-?\d+\.\d+) Depth (\d+\.\d+)', line)
            if match:
                name, lat, long, depth = match.groups()
                data.append({'Name': name, 'Lat': float(lat), 'Long': float(long), 'Depth': float(depth)})
    return data

data = extract_info_from_file('../phase1/catalog.reference.txt')

df = pd.DataFrame(data)
print(df)
df.to_csv('data/reference_data.csv', index=False)
