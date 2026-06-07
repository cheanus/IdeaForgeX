# 从 neo4j 容器中迁移数据

## 导出数据

使用新容器挂载卷来导出数据：

```bash
docker run --rm \ 
  -v ideaforgex_neo4j_personal_data:/data \
  -v /tmp:/tmp \     
  neo4j:community \
  neo4j-admin database dump neo4j \
  --to-path=/tmp
```

这会将数据库导出为 `/tmp/neo4j.dump`，你可以在主机的 `/tmp` 目录下找到它。

## 上传数据到 Aura

将数据复制到容器内
```bash
docker cp /tmp/neo4j.dump ideaforgex-neo4j-personal:/var/lib/neo4j/import/
```

进入容器
```bash
docker exec -it ideaforgex-neo4j-personal bash
```

上传数据至 Aura
```bash
bin/neo4j-admin database upload neo4j --from-path=/var/lib/neo4j/import/ --to-uri=neo4j+s://c800d2a5.databases.neo4j.io --overwrite-destination=true
rm /var/lib/neo4j/import/neo4j.dump
```
