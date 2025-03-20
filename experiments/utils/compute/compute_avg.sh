#!/usr/bin/env bash

echo "====================================="
cat compute_avg.in
echo "====================================="
python compute_avg.py `cat compute_avg.in | grep Time: | tr "Time:" " "`
