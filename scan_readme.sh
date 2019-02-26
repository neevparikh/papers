#!/bin/bash

function make_readme {
    python3 fix_readme.py
}

while true
do
    inotifywait -q -e modify readme.md
    make_readme
done

