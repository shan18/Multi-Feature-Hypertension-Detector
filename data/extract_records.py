import os
import argparse
import random
import json
import pandas as pd

from db_user_info import (
    DB_INFO,
    HEALTHY_BMI_RANGE,
    HEALTHY_BSA_RANGE,
    HEALTHY_SBP_RANGE,
    HEALTHY_DBP_RANGE,
    SBP_DBP_DIFF_RANGE,
)


def parse_ihr(ihr_file):
    user_ihr = []
    # Read from file
    with open(ihr_file) as f:
        user_ihr = [line.strip().split('\t')[1] for line in f]

    # Write correct values to file
    with open(ihr_file, 'w') as f:
        for line in user_ihr:
            f.write(line + '\n')


def parse_user_info(data_dir, db_name):
    df = pd.read_csv(os.path.join(data_dir, DB_INFO[db_name]['user_info_filename']), sep='\t')
    df.drop(columns=['Weight', 'Height', 'IMT MAX', 'LVMi', 'EF'], inplace=True)
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

    # Shuffle the records
    df = df.sample(frac=1)

    return [{**x, 'hypertensive': 1} for x in df.to_dict('records')]


def generate_user_info(data_dir):
    db_info = DB_INFO[os.path.basename(data_dir)]

    records = [os.path.splitext(x)[0] for x in os.listdir(data_dir) if x.endswith('.hea')]
    random.shuffle(records)

    user_info_records = []
    for count, record in enumerate(records):
        # Get gender
        gender = 0 if count < db_info['men'] else 1

        # Get age range
        age_range_key = 'age_range_men' if gender == 0 else 'age_range_women'

        # Get BP
        sbp = random.randint(HEALTHY_SBP_RANGE[0], HEALTHY_SBP_RANGE[1])
        bp_diff = min(max(random.randint(SBP_DBP_DIFF_RANGE[0], SBP_DBP_DIFF_RANGE[1]), HEALTHY_DBP_RANGE[0]), HEALTHY_DBP_RANGE[1])
        dbp = sbp - bp_diff

        # Create record
        user_info_records.append({
            'record': record,
            'gender': gender,
            'age': random.randint(db_info[age_range_key][0], db_info[age_range_key][1] + 1),
            'bsa': random.uniform(HEALTHY_BSA_RANGE[0], HEALTHY_BSA_RANGE[1]),
            'bmi': random.uniform(HEALTHY_BMI_RANGE[0], HEALTHY_BMI_RANGE[1]),
            'smoker': 0,
            'sbp': sbp,
            'dbp': dbp,
            'vascular_event': 0,
            'hypertensive': 0,
        })

    return user_info_records


def extract_ihr(data_dir, ihr_path):
    os.chdir(data_dir)
    ihr_files = [os.path.splitext(x)[0] for x in os.listdir(data_dir) if x.endswith('.hea')]
    db_name = os.path.basename(data_dir)
    format = DB_INFO[db_name]['annotations']

    print('Extracting IHR data from', db_name)
    total_files = len(ihr_files)
    print(f'\rProgress: 0/{total_files}', end='\r')
    for count, ihr_file in enumerate(ihr_files):
        os.system(f'ihr -r {ihr_file} -a {format} > "{os.path.join(ihr_path, ihr_file)}.txt"')
        parse_ihr(f'{os.path.join(ihr_path, ihr_file)}.txt')
        print(f'\rProgress: {count + 1}/{total_files}', end='\r')
    print('\n')


def extract_user_info(data_dir):
    db_name = os.path.basename(data_dir)
    print('Extracting user info from', db_name)
    return (
        parse_user_info(data_dir, db_name)
        if 'user_info_filename' in DB_INFO[db_name]
        else generate_user_info(data_dir)
    )


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dbs', '--databases', nargs='+',
        default=[
            os.path.join(BASE_DIR, 'files', 'shareedb'),
            os.path.join(BASE_DIR, 'files', 'nsrdb'),
            os.path.join(BASE_DIR, 'files', 'nsr2db'),
        ], help='Path to databases for extracting IHR data'
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
        for db in args.databases:
            extract_ihr(db, args.ihr_path)

    print('Storing user info')
    user_records = []
    for db in args.databases:
        user_records += extract_user_info(db)
    with open(args.info, 'w') as f:
        json.dump(user_records, f, indent=2)
    print('Done.')
