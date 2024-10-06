#!/bin/bash

min_duration=999999999
max_duration=0
min_file=""
max_file=""

for file in ${1}/*.wav
do
    duration=$(soxi -D "$file")
    if (( $(echo "$duration < $min_duration" | bc -l) )); then
        min_duration=$duration
        min_file=$file
    fi
    if (( $(echo "$duration > $max_duration" | bc -l) )); then
        max_duration=$duration
        max_file=$file
    fi
done

echo "Min duration: $min_duration seconds, File: $min_file"
echo "Max duration: $max_duration seconds, File: $max_file"