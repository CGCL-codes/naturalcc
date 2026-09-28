# 测试主题：SQL 数据库架构设计

这是一个用于测试 coding agent 数据库设计能力的项目。

## 测试目标
- 测试 agent 是否能正确设计范式化的数据模型
- 测试 agent 是否能正确建立表关系（外键、索引）
- 测试 agent 是否能正确实现视图、触发器、存储过程
- 测试 agent 是否能正确考虑性能（索引、查询优化）

## 项目结构
```
project_09_sql_database/
├── README.md
├── schema/
│   ├── 01_create_tables.sql     # 建表语句
│   ├── 02_create_indexes.sql    # 索引
│   ├── 03_create_views.sql      # 视图
│   ├── 04_seed_data.sql         # 测试数据
│   └── 05_queries.sql           # 常用查询
└── docs/
    └── ER_diagram.md            # ER 图说明
```
