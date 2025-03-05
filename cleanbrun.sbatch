#!/bin/bash

# Remove directories and files matching the patterns in parallel within brun* directories
find brun* -type d -name "Model" -exec rm -rf {} + &
find brun* -type d -name "Method" -exec rm -rf {} + &
find brun* -type f -name "Systemsetup.png" -exec rm -f {} + &
find brun* -type d -name "InitialStates" -exec rm -rf {} + &
find brun* -type f -name "input.txt" -exec rm -f {} + &
find brun* -type f -name "Rd*.npy" -exec rm -f {} + &
find brun* -type f -name "serial.py" -exec rm -f {} + &
find brun* -type f -name "error*.err" -exec rm -f {} + &

# Wait for all background processes to finish
wait

echo "Cleanup complete."