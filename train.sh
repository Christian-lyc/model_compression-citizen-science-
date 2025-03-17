nohup python3 -u fine_tune.py >unstr20.out 2>&1 &

job_pid=$!

echo "PID of the job: $job_pid"
