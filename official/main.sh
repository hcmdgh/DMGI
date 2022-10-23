#!/bin/bash 

set -eux 

if [ ! -f main.py ]; then
    cd official 
fi

python main.py 
