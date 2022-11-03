import os
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="results.csv", type=str, help="filename of the concatenated csv.")
    parser.add_argument("--csv_dir", default="./csvs", type=str, help="Directory where the csv's are saved.")
    parser.add_argument("--sep", default=",", type=str, help="csv seperator.")
    parser.add_argument("--enc", default="utf-8", type=str, help="Encoding.")
    args = parser.parse_args()
    
    out_frame = pd.DataFrame()
    for elem in tqdm(os.listdir(args.csv_dir)):
        if not elem.endswith(".csv"):
            continue
        file =  args.csv_dir + "/" + elem
        try:
            df = pd.read_csv(file, sep=args.sep, header=0, encoding=args.enc)
        except UnicodeDecodeError:
            print(file)
            raise
        out_frame = out_frame.append(df, ignore_index=True)
    print(out_frame)
    out_frame.to_csv(args.file, sep=args.sep, encoding=args.enc)


if __name__ == "__main__":
    main()