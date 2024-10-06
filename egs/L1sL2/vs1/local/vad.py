# Copyright (c) [2024] Haopeng Geng, University of Tokyo
# MIT License (https://opensource.org/licenses/MIT)

# For long duration audio, VAD the start and end timestamp and make into segments

from rVADfast.process import rVADfast_multi_process
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--db_root', help='Root directory of the database')
    parser.add_argument('--scp_file', help='Path to scp file')
    parser.add_argument('--vad_dir', help='Path to save vad files')
    parser.add_argument('--segments_path', help='Path to savesegments with start and end only')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers (default: 8)')
    args = parser.parse_args()

    # MKDIR if vad_dir does not exist
    # if vad_dir exists, clean it and create a new one
    import os
    if not os.path.exists(args.vad_dir):
        os.makedirs(args.vad_dir)
    else:
        os.system('rm -r ' + args.vad_dir)
        os.makedirs(args.vad_dir)
        
    # get all wav files from scp file
    with open(args.scp_file, 'r') as f:
        lines = f.readlines()
        ids = [line.split()[0] for line in lines]
        files = [line.split()[1] for line in lines]
    # TODO check if the file exists in the db_root base on scp file
    from pathlib import Path
    
    db_root = Path(files[0]).parent
    print(db_root)
    # save vad files in vad_dir
    # Assert all files are saved in db_root
    rVADfast_multi_process(db_root, save_folder=args.vad_dir, extension='wav', n_workers=args.n_workers, trim_non_speech=False)

    # if segments file exists, clean it and create a new one
    if os.path.exists(args.segments_path):
        os.system('rm ' + args.segments_path)
    with open(args.segments_path, 'w') as f_segments:
        for i in range(len(ids)):
            with open(Path(args.vad_dir) / Path(ids[i] + '_vad.txt'), 'r') as f:
                f.readline() # skip header
                start = f.readline().split(',')[0] # start time fo first segment
                # remove \n for start
                if start[-1] == '\n':
                    start = start[:-1]
                end = f.readline().split(',')[-1] # end time of last segment
                # print(start, end)
                # save segments with start and end only in segments_path, format: <id> <id> <start> <end>
                f_segments.write(ids[i] + ' ' + ids[i] + ' ' + start + ' ' + end)
                
    print('Done!')
if __name__ == '__main__':
    main()

main()