import re
from pprint import pprint


def parse_geographic(value):
    parts = value.split()
    return {
        'ot_year': parts[1],
        'ot_month': parts[2],
        'ot_day': parts[3],
        'ot_hour': parts[4],
        'ot_minute': parts[5],
        'ot_second': parts[6],
        'lat': parts[8],
        'long': parts[10],
        'depth': parts[12]
    }


def parse_hypocenter(value):
    parts = value.split()
    return {
        'x': parts[1],
        'y': parts[3],
        'z': parts[5],
        'ot': parts[7]
    }


def parse_quality(value):
    parts = value.split()
    return {
        'pmax': parts[1],
        'mfmin': parts[3],
        'mfmax': parts[5],
        'rms': parts[7],
        'nphs': parts[9],
        'gap': parts[11],
        'dist': parts[13],
        'mamp': parts[15],
        'mdur': parts[17]
    }


def parse_vpvsratio(value):
    parts = value.split()
    return {
        'VpVsRatio': parts[0],
        'Npair': parts[1],
        'Diff': parts[2]
    }


def parse_statistics(value):
    parts = value.split()
    return {
        'ExpectX': parts[0],
        'ExpectY': parts[1],
        'ExpectZ': parts[2],
        'CovXX': parts[3],
        'CovXY': parts[4],
        'CovXZ': parts[5],
        'CovYY': parts[6],
        'CovYZ': parts[7],
        'CovZZ': parts[8],
        'EllAz1': parts[9],
        'Dip1': parts[10],
        'Len1': parts[11],
        'Az2': parts[12],
        'Dip2': parts[13],
        'Len2': parts[14],
        'Len3': parts[15]
    }


def parse_stat_geog(value):
    parts = value.split()
    return {
        'ExpectLat': parts[0],
        'ExpectLong': parts[1],
        'ExpectDepth': parts[2]
    }


def parse_info(info):
    if 'GEOGRAPHIC' in info:
        print(info['GEOGRAPHIC'])
        info['GEOGRAPHIC'] = parse_geographic(info['GEOGRAPHIC'])
    if 'HYPOCENTER' in info:
        info['HYPOCENTER'] = parse_hypocenter(info['HYPOCENTER'])
    if 'QUALITY' in info:
        info['QUALITY'] = parse_quality(info['QUALITY'])
    if 'VPVSRATIO' in info:
        info['VPVSRATIO'] = parse_vpvsratio(info['VPVSRATIO'])
    if 'STATISTICS' in info:
        info['STATISTICS'] = parse_statistics(info['STATISTICS'])
    if 'STAT_GEOG' in info:
        info['STAT_GEOG'] = parse_stat_geog(info['STAT_GEOG'])
    # 添加更多的解析函数调用...

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
                    parse_info(current_info)
                    extracted_info.append(current_info)
                current_info = {}
            else:
                match = re.match(r'(\w+)\s+(.*)', line)
                if match:
                    key, value = match.groups()
                    current_info[key] = value
    if current_info:
        parse_info(current_info)
        extracted_info.append(current_info)
    return extracted_info


extracted_info = extract_info_from_file('../raw_data/mayotte.sum.grid0.loc.hyp.bias')
pprint(extracted_info[0])
