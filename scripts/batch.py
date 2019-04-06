import argparse
import os

import pandas as pd

from main import main, valid_file, valid_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mineral prediction in"
                                                 "batch mode with a CSV.")
    parser.add_argument('csv', type=valid_file)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, skipinitialspace=True)

    required = ['standards_dir', 'meteorite_dir', 'target_minerals_file', 'output_dir']
    convert_to_int = ['n', 'unknown_n', 'batch_size']
    for col in required:
        if col not in df.columns:
            raise ValueError('Required column `%s` not in CSV' % col)

    for col in ['meteorite_dir', 'target_minerals_file', 'output_dir', 'mask']:
        if (col in df.columns) and df[col].str.contains(',').any():
            raise ValueError('%s must be a single value (must not contain `,`)' % col)

    df2 = df[['meteorite_dir', 'target_minerals_file', 'output_dir']]
    for col in df.columns:
        if col in df2:
            continue

        series = df[col].astype(str).str.strip().str.split(
            ',', expand=True
        ).stack().rename(col).str.strip().reset_index(level=1, drop=True)

        if col in convert_to_int:
            series = series.astype(int)
        df2 = df2.join(series)

    count = 0
    for i, row in df2.iterrows():
        args = [
            valid_dir(row['standards_dir']), valid_dir(row['meteorite_dir']),
            valid_file(row['target_minerals_file']), os.path.join(row['output_dir'], str(count))
        ]
        kwargs = {
            k: row[k]
            for k in row.index if (
                (k not in required) and (not pd.isnull(row[k])) and row[k] != 'nan'
            )
        }

        main(*args, **kwargs)
        count += 1
