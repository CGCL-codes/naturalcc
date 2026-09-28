# 测试主题：Python FastAPI REST API 设计

这是一个用于测试 coding agent 创建/修改 Python Web API 项目的项目。

## 测试目标
- 测试 agent 是否能正确添加新的 API 路由
- 测试 agent 是否能正确处理 Pydantic 数据模型
- 测试 agent 是否能正确处理异步数据库调用
- 测试 agent 是否能正确处理错误和异常

## 项目结构
```
project_01_python_api/
├── README.md
├── main.py              # FastAPI 应用入口
├── models.py            # 数据模型
├── database.py          # 数据库连接
├── routes/
│   ├── __init__.py
│   ├── users.py         # 用户相关路由
│   └── items.py         # 商品相关路由
└── requirements.txt     # 依赖
```
