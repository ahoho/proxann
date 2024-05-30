#!/bin/bash

URL="https://github.com/mimno/Mallet/releases/download/v202108/Mallet-202108-bin.tar.gz"

OUTPUT_DIR="src/train"
OUTPUT_FILE="${OUTPUT_DIR}/Mallet-202108-bin.tar.gz"

mkdir -p $OUTPUT_DIR

wget -O $OUTPUT_FILE $URL

if [ $? -eq 0 ]; then
    echo "Download completed successfully and saved to ${OUTPUT_FILE}."

    tar -xzvf $OUTPUT_FILE -C $OUTPUT_DIR

    if [ $? -eq 0 ]; then
        echo "Unzipped successfully to ${OUTPUT_DIR}."

        rm $OUTPUT_FILE
        if [ $? -eq 0 ]; then
            echo "Removed tar file ${OUTPUT_FILE}."
        else
            echo "Failed to remove tar file ${OUTPUT_FILE}."
        fi
    else
        echo "Unzipping failed."
    fi
else
    echo "Download failed."
fi