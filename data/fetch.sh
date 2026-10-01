#!/bin/sh
# Fetch the benchmark dataset: gas turbine CO and NOx emissions,
# UCI Machine Learning Repository, dataset 551, CC BY 4.0.
# 36,733 hourly readings, 2011-2015. Writes gt_2011.csv .. gt_2015.csv here.
set -e
cd "$(dirname "$0")"
URL=https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip
curl -sL -o gt.zip "$URL"
unzip -o -q gt.zip
rm -f gt.zip
echo "fetched: $(ls gt_*.csv | tr '\n' ' ')"
