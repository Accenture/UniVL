#!/bin/bash
SRC="$UNIVL_DATA_ROOT/laion-aes"
DST="$UNIVL_DATA_ROOT/laion-aes"
mkdir -p "$DST"

for tar in "$SRC"/*.tar; do
    name=$(basename "$tar" .tar)
    if [ -d "$DST/$name" ]; then
        echo "SKIP $name (exists)"
        continue
    fi
    mkdir -p "$DST/$name"
    tar xf "$tar" -C "$DST/$name" && echo "OK $name" || echo "FAIL $name"
done
echo "DONE. Total dirs: $(ls -d $DST/*/ | wc -l)"
