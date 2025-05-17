from celery import Celery

app = Celery(
    "autorag",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
    include=["tasks.trial_tasks"],
)

# Celery 설정
app.conf.update(
    broker_url="redis://redis:6379/0",
    result_backend="redis://redis:6379/0",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
)
