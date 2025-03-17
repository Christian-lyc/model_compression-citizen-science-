#nohup python3 -u main.py >log.out 2>&1 &
nohup python3 -u testonnx.py >log.out 2>&1 &

job_pid=$!

echo "PID of the job: $job_pid"
