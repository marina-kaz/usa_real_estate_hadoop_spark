
```
docker compose build
```

```
docker compose up -d
docker compose up -d --scale spark-yarn-worker=3
```

```
docker exec da-spark-yarn-master spark-submit --master yarn --deploy-mode cluster ./apps/client.py
```

```
docker compose down
```