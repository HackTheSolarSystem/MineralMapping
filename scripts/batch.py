import argparse
import os
import traceback

import pandas as pd

from main import main, valid_file, valid_dir, valid_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mineral prediction in"
                                                 "batch mode with a CSV.")
    parser.add_argument('csv', type=valid_file)
    parser.add_argument('--batch_summary_csv', type=str, default=None,
                        help="An optional output CSV with mineral counts for "
                        "all of the runs.")
    args = parser.parse_args()
    batch_summary_csv = args.batch_summary_csv

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
    results = []
    for col in df.columns:
        if col in df2:
            continue

        series = df[col].astype(str).str.strip().str.split(
            ',', expand=True
        ).stack().rename(col).str.strip().reset_index(level=1, drop=True)

        if col in convert_to_int:
            series = series.astype(float)
        df2 = df2.join(series)

    count = 0
    for i, row in df2.iterrows():
        args = [
            valid_dir(row['standards_dir']), valid_dir(row['meteorite_dir']),
            valid_file(row['target_minerals_file']), os.path.join(row['output_dir'], str(count))
        ]
        kwargs = {
            k: valid_model(row[k]) if k == 'model' else row[k]
            for k in row.index if (
                (k not in required) and (not pd.isnull(row[k])) and
                row[k] != 'nan' and not pd.isnull(row[k])
            )
        }
        try:
            results.append(main(*args, **kwargs))
        except:
            print('The following exception occured with the parameters:')
            print(args)
            print(kwargs)
            traceback.print_exc()

        count += 1

    if batch_summary_csv:
        df = pd.concat(results, sort=True)
        cols = ['path', 'mask']
        df = df[cols + [c for c in df.columns if c not in cols]]

        df.to_csv(batch_summary_csv, index=False)
