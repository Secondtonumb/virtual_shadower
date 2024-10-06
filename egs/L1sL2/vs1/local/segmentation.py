# Copyright (c) [2024] Haopeng Geng, University of Tokyo
# MIT License (https://opensource.org/licenses/MIT)

import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--db_root', help='Root directory of the database')
    parser.add_argument('--scp_file', help='Path to scp file')
    parser.add_argument('--segments_path', help='Path to savesegments with start and end only')
    args = parser.parse_args()
    
    # get all wav files from scp file
    with open(args.scp_file, 'r') as f:
        lines = f.readlines()
        ids = [line.split()[0] for line in lines]
        files = [line.split()[1] for line in lines]
    # TODO check if the file exists in the db_root base on scp file
    from pathlib import Path
    
    # get segment.txt
    # save record into segments_path, format: <id> <id> <start> <end>
    segment_cnt = []
    for file in files:
        segment_file = Path(file).parent / Path('segment.txt')
        with open(segment_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                utt_id, start, end = line.split()[0], line.split()[1], line.split()[2]
                record = f"{utt_id[-9:]} {Path(file).stem} {round(float(start), 2)} {round(float(end), 2)}\n"
                # print(utt_id[-9:], Path(file).stem, start, end)
                segment_cnt.append(record)
    with open(args.segments_path, 'w') as f:
        f.writelines(segment_cnt)
    # import pdb; pdb.set_trace()
main()