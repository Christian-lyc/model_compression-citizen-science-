nohup python3 -u str_testonnx.py >log.out 2>&1 &
#nohup python3 -u stru_test.py >log.out 2>&1 &

job_pid=$!

echo "PID of the job: $job_pid"
