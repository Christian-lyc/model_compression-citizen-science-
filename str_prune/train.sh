nohup python3 -u stru_prune.py --model '/export/home/yliu/runs/detect/train2/weights/best.pt' --target-prune-rate 0.4 >log.out 2>&1 &

job_pid=$!

echo "PID of the job: $job_pid"
