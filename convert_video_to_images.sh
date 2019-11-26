#!/bin/bash

# Reference - https://github.com/bityangke/3d-DenseNet/blob/master/data_prepare/convert_video_to_images.sh 
## To run ./convert_video_to_images.sh folder_name

for folder in $1/*
do
    for file in "$folder"/*.mp4
    do
        if [[ ! -d "${file[@]%.mp4}" ]]; then
            mkdir -p "${file[@]%.mp4}"
        fi
        ffmpeg -i "$file" -vf fps=30 "${file[@]%.mp4}"/%05d.jpg
        rm "$file"
    done
done