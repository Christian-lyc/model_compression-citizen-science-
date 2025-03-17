nohup python3 -u prune_test.py >log.out 2>&1 &

job_pid=$!

echo "PID of the job: $job_pid"
