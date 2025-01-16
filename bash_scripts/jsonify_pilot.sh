#!/bin/bash

# Navigate to the script's directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

# Current working directory
pwd

python3 get_user_study_data.py --config "config/user_study/config_test.conf"
