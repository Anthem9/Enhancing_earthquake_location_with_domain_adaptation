import re
import pandas as pd


def parse_geographic(value):
    parts = value.split()
    return {
        'lat': float(parts[8]),
        'long': float(parts[10]),
        'depth': float(parts[12])
    }


def parse_signature(value):
    match = re.search(r'obs:.*/([^/]*)\.obs', value)
    if match:
        return {'Name': match.group(1)[-6:]}
    else:
        return {}

def parse_confidence_ellipsoid(value):
    parts = value.split()
    return {
        'semiMajorAxisLength': float(parts[1]),
        'semiMinorAxisLength': float(parts[3]),
        'semiIntermediateAxisLength': float(parts[5]),
        'majorAxisPlunge': float(parts[7]),
        'majorAxisAzimuth': float(parts[9]),
        'majorAxisRotation': float(parts[11])
    }


def parse_search(value):
    parts = value.split()
    # print(parts[-1])

    return {'scatter_volume': float(parts[-1])}


def parse_info(info):
    result = {}
    if 'GEOGRAPHIC' in info:
        result.update(parse_geographic(info['GEOGRAPHIC']))
    if 'SIGNATURE' in info:
        # print(parse_signature(info['SIGNATURE']))
        result.update(parse_signature(info['SIGNATURE']))
    if 'QML_ConfidenceEllipsoid' in info:
        # print(info['QML_ConfidenceEllipsoid'])
        result.update(parse_confidence_ellipsoid(info['QML_ConfidenceEllipsoid']))
    if 'SEARCH' in info:
        # print(info['SEARCH'])
        result.update(parse_search(info['SEARCH']))
    return result


def extract_info_from_file(file_name):
    extracted_info = []
    current_info = {}
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('NLLOC'):
                if current_info:
                    extracted_info.append(parse_info(current_info))
                current_info = {}
            else:
                match = re.match(r'(\w+)\s+(.*)', line)
                if match:
                    key, value = match.groups()
                    current_info[key] = value
    if current_info:
        extracted_info.append(parse_info(current_info))
    return extracted_info


extracted_info = extract_info_from_file('raw_data/mayotte.sum.grid0.loc.hyp.bias')
df = pd.DataFrame(extracted_info)
name = df.pop('Name')  # remove column Name and store it in name
df.insert(0, 'Name', name)  # insert column Name with its values at the first position

df.to_csv('biased_data.csv', index=False)
print(df)
