#!/bin/bash

# Recursive script to touch all files in /scratch-shared/igodzwon/Project_AI
# This updates the access/modification time of all files

SCRATCH_DIR="/scratch-shared/igodzwon/Project_AI"

if [ ! -d "$SCRATCH_DIR" ]; then
    echo "Error: Directory $SCRATCH_DIR does not exist"
    exit 1
fi

echo "Touching all files in $SCRATCH_DIR..."
echo "This may take a while depending on the number of files..."

# Find all files recursively and touch them
find "$SCRATCH_DIR" -type f -exec touch {} \;

echo "Done! All files have been touched."

