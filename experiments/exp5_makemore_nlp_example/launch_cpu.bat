start /affinity 2 python makemore.py --type mlp --batch-size 64 --n-embd2 1024 --num-workers 0 --max-steps 4000 --device cpu -i names.txt -o names
