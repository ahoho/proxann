#!/bin/bash

LOG_FILE="data/files_pilot/run_user_study.log"
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

##############
# q1_then_q3 #
##############
log "*** Running: python3 run_user_study.py --model_type chatgpt-4o-latest,llama_cpp,llama3.2 --prompt_mode q1_then_q3"
python3 run_user_study.py --model_type chatgpt-4o-latest,llama_cpp,llama3.2 --prompt_mode q1_then_q3 2>&1 | tee -a "$LOG_FILE"
if [ $? -eq 0 ]; then
    log "q1_then_q3 completed"
else
    log "Error in prompt_mode q1_then_q3"
fi

##############
# q1_and_q3 #
##############
log "*** Running: python3 run_user_study.py --model_type chatgpt-4o-latest,llama_cpp,llama3.2 --prompt_mode q1_and_q3"
python3 run_user_study.py --model_type chatgpt-4o-latest,llama_cpp,llama3.2 --prompt_mode q1_and_q3 2>&1 | tee -a "$LOG_FILE"
if [ $? -eq 0 ]; then
    log "q1_and_q3 completed"
else
    log "Error in prompt_mode q1_and_q3"
fi

log "Script execution completed"
