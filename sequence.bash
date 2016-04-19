#!/bin/bash
COUNTER=0
while [ $COUNTER -lt 19 ]; do
    ./runYarn.bash kmeans_subject.py $COUNTER 2>logs/kmeans_subject"$COUNTER".log
    wait
    let COUNTER=COUNTER+1
done

