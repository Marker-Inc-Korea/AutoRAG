from celery import Celery

app = Celery('autorag',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/0',
             include=['tasks.trial_tasks'])

# Celery 설정
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)