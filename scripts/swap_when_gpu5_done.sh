#!/bin/bash
# Watcher for P2 launcher: when GPU 5 finishes all 123 of seed 2022's tasks,
# kill the launcher and restart with the remaining work redistributed to GPU 5
# (--gpus 5 5 6). Logs every poll to logs/swap_watcher.log.
#
# Spawn in its own tmux session:
#   tmux new-session -d -s p2-watcher -c /home/xinkaiz/VGLG \
#       'bash scripts/swap_when_gpu5_done.sh'
#
# Abort cleanly:
#   tmux kill-session -t p2-watcher

set -u

ROOT=/home/xinkaiz/VGLG
LOG=$ROOT/logs/p2_master.log
WATCH=$ROOT/logs/swap_watcher.log
PY=/data/xinkaiz/conda_envs/vglg/bin/python
TARGET=123
POLL_INTERVAL=300   # 5 min

ts() { date '+%Y-%m-%d %H:%M:%S'; }

mkdir -p "$(dirname "$WATCH")"
echo "$(ts) Watcher started. Polling every ${POLL_INTERVAL}s for GPU 5 to reach $TARGET resolved tasks." > "$WATCH"

while true; do
    if [ ! -f "$LOG" ]; then
        echo "$(ts) ERROR: master log $LOG not found, sleeping." >> "$WATCH"
        sleep $POLL_INTERVAL
        continue
    fi
    # Count GPU 5's resolved tasks: any of OK / SKIP / FAIL / OOM.
    n=$(grep -cE '^\[GPU5\][[:space:]]+[0-9]+/[0-9]+ (OK|SKIP|FAIL|OOM)[[:space:]]' "$LOG" 2>/dev/null || echo 0)
    n=${n:-0}
    started=$(grep -cE '^\[GPU5\][[:space:]]+[0-9]+/[0-9]+ START' "$LOG" 2>/dev/null || echo 0)
    started=${started:-0}
    echo "$(ts) GPU5 resolved=$n started=$started (target $TARGET)" >> "$WATCH"

    if [ "$n" -ge "$TARGET" ]; then
        echo "$(ts) *** TRIGGER: GPU 5 has resolved $n tasks (>= $TARGET). Initiating swap. ***" >> "$WATCH"

        # 1. Kill the existing launcher (tmux session p2-main)
        tmux kill-session -t p2-main 2>>"$WATCH"
        echo "$(ts) Killed tmux session p2-main." >> "$WATCH"
        sleep 5

        # 2. Force-kill any straggling trainer subprocess (subprocess.run children of dead launcher)
        pkill -9 -f "src.train.trainer" 2>>"$WATCH"
        echo "$(ts) Killed straggling trainer processes." >> "$WATCH"
        sleep 10

        # 3. Restart launcher with seeds redistributed: GPU 5 takes 2021+2022, GPU 6 keeps 2023.
        #    --rerun-failed retries the OOMs (so e.g. GPU 4's OOM gets a fresh shot on GPU 5).
        tmux new-session -d -s p2-main -c "$ROOT" \
            "$PY scripts/run_p2_main.py --gpus 5 5 6 --rerun-failed 2>&1 | tee -a logs/p2_master.log"
        echo "$(ts) Restarted launcher in tmux p2-main with --gpus 5 5 6 --rerun-failed." >> "$WATCH"
        echo "$(ts) Watcher done. Exiting." >> "$WATCH"
        exit 0
    fi
    sleep $POLL_INTERVAL
done
