# 从 neo4j 容器中导出数据
打开浏览器访问 http://localhost:7475，登录 Neo4j 浏览器界面。

在左侧输入框中输入以下 Cypher 查询，导出所有节点和关系：

```cypher
CALL apoc.export.csv.all("neo4j_export.csv", {
  format: "plain",
  quotes: "always",
  useTypes: true
});
```

执行后会在 Neo4j 容器的 `/var/lib/neo4j/import` 目录下生成 `neo4j_export.csv` 文件。你可以使用以下命令将其复制到主机：

```bash
docker cp ideaforgex-neo4j-personal:/var/lib/neo4j/import/neo4j_export.csv ./neo4j_export.csv
```

这样就得到了包含所有节点和关系的 CSV 文件，可以使用 Excel、Pandas 等工具进行分析和处理，也可以导入到另一个 Neo4j 实例中进行查询和可视化。

之后可以删除容器中的导出文件：

```bash
docker exec ideaforgex-neo4j-personal rm /var/lib/neo4j/import/neo4j_export.csv
```
