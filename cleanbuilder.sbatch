#!/bin/bash

# Remove all *.qjob files
find . -type f -name "*.qjob" -exec rm -f {} \;

# Remove all *_batch_* files
find . -type f -name "*_batch_*" -exec rm -f {} \;

echo "Cleanup complete."