import os
import json
import argparse


def read_user_info(path):
    with open(path) as f:
        user_info = json.load(f)
    return user_info


def get_ihr_sequences(ihr_file, sequence_limit=4096):
    with open(ihr_file) as f:
        ihr_values = [float(line.strip()) for line in f]
    return [
        ihr_values[index:index + sequence_limit]
        for index in range(0, len(ihr_values), sequence_limit)
        if len(ihr_values) - index > sequence_limit
    ]


def create_data(user_info, ihr_dir, output_file):
    data = []
    num_records = len(user_info)
    print('Creating data...')
    print(f'\rProgress: 0/{num_records}', end='\r')
    for count, record in enumerate(user_info):
        ihr_sequences = get_ihr_sequences(os.path.join(ihr_dir, f'{record["record"]}.txt'))
        data.extend([{
            'ihr': sequence,
            'gender': record['gender'],
            'age': record['age'],
            'hypertensive': record['hypertensive'],
        } for sequence in ihr_sequences])
        print(f'\rProgress: {count + 1}/{num_records}', end='\r')
    print()

    print('Saving...')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print('Done.')


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--user_info', default=os.path.join(BASE_DIR, 'user_info.json'),
        help='Path to user info json'
    )
    parser.add_argument(
        '--ihr_dir', default=os.path.join(BASE_DIR, 'files', 'ihr_data'),
        help='Path to folder containing the IHR data'
    )
    parser.add_argument(
        '--output', default=os.path.join(BASE_DIR, 'dataset.json'),
        help='Name of file in which dataset will be stored'
    )
    args = parser.parse_args()

    create_data(
        read_user_info(args.user_info),
        args.ihr_dir,
        args.output,
    )
