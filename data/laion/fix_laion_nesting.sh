#!/bin/bash
# Fix tar extraction nesting: laion-aes/00000/00000/* → laion-aes/00000/*
# Moves contents of inner dir up, then removes empty inner dir.

BASE="$UNIVL_DATA_ROOT/laion-aes"

for dir in "$BASE"/*/; do
    name=$(basename "$dir")
    inner="$dir/$name"
    if [ -d "$inner" ]; then
        # Move all contents from inner to outer
        mv "$inner"/* "$dir/" 2>/dev/null
        mv "$inner"/.* "$dir/" 2>/dev/null
        rmdir "$inner" 2>/dev/null
        echo "OK $name"
    else
        echo "SKIP $name (no inner dir)"
    fi
done
echo "DONE"
