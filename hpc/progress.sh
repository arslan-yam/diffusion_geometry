#!/bin/bash
# What is running, how many runs each results file has, and the last line of every log.
cd "$(dirname "$0")/.."
squeue -u $USER -o "%.10i %.18j %.2t %.10M %.12l %R"
echo; for f in results/*.jsonl; do [ -f "$f" ] && printf "%-40s %3d runs\n" "$f" "$(wc -l < "$f")"; done
for f in results/exp3_*.json; do [ -f "$f" ] && printf "%-40s %s\n" "$f" "$(python3 -c "import json;print(len(json.load(open('$f'))['records']),'images')")"; done
echo; for f in $(ls -t logs/*.out 2>/dev/null | head -25); do printf "%-32s %s\n" "$(basename $f)" "$(grep -v '^\s*$' $f | grep -v -i 'elapsed\|maximum resident\|Command being' | tail -1 | cut -c1-110)"; done
