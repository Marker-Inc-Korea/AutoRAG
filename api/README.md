# AutoRAG-API
Quart API server for running AutoRAG and various data creations.

## 2.사용 방법:
Docker Compose로 전체 스택 실행:
```
docker-compose up -d
```
### 2. 모니터링:
Flower UI: http://localhost:5555
Redis Commander: http://localhost:8081

### Test:
```
python -m pytest tests/test_trial_config.py -v
python -m pytest tests/test_app.py -v



```

### 4. 프로젝트 생성
```
cd ..
docker-compose up
```


```
# cd api
python -m celery -A celery_app worker --loglevel=INFO
```
