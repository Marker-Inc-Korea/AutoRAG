#!/bin/bash

# Start API server with standard event loop
uvicorn app:app --host 0.0.0.0 --port 5000 --loop asyncio --reload &
# --reload-dir /app/api &

# Start Celery worker
# multiprocessing을 사용하지 않으므로 daemon 프로세스 문제를 피할 수 있음
python -m celery -A celery_app worker --loglevel=INFO  --pool=solo # --concurrency=1

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
