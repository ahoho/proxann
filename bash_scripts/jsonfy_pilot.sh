#!/bin/bash

# Navigate to the script's directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

# Current working directory
pwd

python3 get_info_models.py --config "config/config_pilot.conf"
