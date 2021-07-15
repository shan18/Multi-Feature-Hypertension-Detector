import os
import argparse
import random
import json
import pandas as pd


def parse_ihr(ihr_file):
    user_ihr = []
    # Read from file
    with open(ihr_file) as f:
        for line in f:
            user_ihr.append(line.strip().split('\t')[1])
    
    # Write correct values to file
    with open(ihr_file, 'w') as f:
        for line in user_ihr:
            f.write(line + '\n')


def extract_ihr(data_dir, ihr_path):
    os.chdir(data_dir)
    ihr_files = [os.path.splitext(x)[0] for x in os.listdir(data_dir) if x.endswith('.hea')]
    format = 'ecg' if data_dir.endswith('nsr2db') else 'qrs'
    
    print('Extracting IHR data from', data_dir)
    total_files = len(ihr_files)
    print(f'\rProgress: 0/{total_files}', end='\r')
    for count, ihr_file in enumerate(ihr_files):
        os.system(f'ihr -r {ihr_file} -a {format} > {os.path.join(ihr_path, ihr_file)}.txt')
        parse_ihr(f'{os.path.join(ihr_path, ihr_file)}.txt')
        print(f'\rProgress: {count + 1}/{total_files}', end='\r')
    print('\n')


def parse_shareedb_user_info(info_path):
    df = pd.read_csv(info_path, sep='\t')
    df.drop(columns=['IMT MAX', 'LVMi', 'EF'], inplace=True)
    df.columns = df.columns.str.lower()
    df.rename(columns={'vascular event': 'vascular_event'}, inplace=True)
    df.record = '0' + df.record.astype(str)
    
    # Fill NaN values
    for column in df.columns:
        df[column] = df[column].fillna(-1)
    
    # Change boolean values
    df.loc[df.smoker == 'yes', 'smoker'] = 1
    df.loc[df.smoker == 'no', 'smoker'] = 0
    df.loc[df.vascular_event != 'none', 'vascular_event'] = 1
    df.loc[df.vascular_event == 'none', 'vascular_event'] = 0
    df.loc[df.gender != 'M', 'gender'] = 1
    df.loc[df.gender == 'M', 'gender'] = 0

    return [{**x, 'hypertensive': 1} for x in df.to_dict('records')]


def parse_nsr2db_user_info(data_dir):
    record = [os.path.splitext(x)[0] for x in os.listdir(data_dir) if x.endswith('.hea')]
    random.shuffle(record)
    return [
        {'record': x, 'gender': 0, 'age': random.randint(29, 76), 'hypertensive': 0}
        for x in record[:30]
    ] + [
        {'record': x, 'gender': 1, 'age': random.randint(58, 73), 'hypertensive': 0}
        for x in record[30:]
    ]


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--shareedb', default=os.path.join(BASE_DIR, 'files', 'shareedb'),
        help='Path to shareedb data'
    )
    parser.add_argument(
        '--nsr2db', default=os.path.join(BASE_DIR, 'files', 'nsr2db'),
        help='Path to nsr2db data'
    )
    parser.add_argument(
        '--shareedb_info', default=os.path.join(BASE_DIR, 'files', 'shareedb_info.txt'),
        help='Path to file containing user record'
    )
    parser.add_argument(
        '--ihr_path', default=os.path.join(BASE_DIR, 'files', 'ihr_data'),
        help='Directory in which RR data will be stored as text'
    )
    parser.add_argument(
        '--info', default=os.path.join(BASE_DIR, 'user_info.json'),
        help='Path where extracted user info will be stored'
    )
    args = parser.parse_args()

    if not os.path.exists(args.ihr_path):
        os.makedirs(args.ihr_path)
        extract_ihr(args.shareedb, args.ihr_path)
        extract_ihr(args.nsr2db, args.ihr_path)

    print('Storing user info')
    with open(args.info, 'w') as f:
        json.dump(
            parse_shareedb_user_info(args.shareedb_info) + parse_nsr2db_user_info(args.nsr2db),
            f, indent=2
        )
    print('Done.')
