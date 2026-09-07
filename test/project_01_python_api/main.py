"""
项目: project_01_python_api
测试主题: Python FastAPI REST API 设计
功能: FastAPI 应用入口，提供用户和商品的 CRUD 接口
"""
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from database import engine, get_db
from models import Base, User, Item
from routes import users, items

# 创建数据库表
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="测试 API",
    description="用于测试 coding agent 的 FastAPI 项目",
    version="1.0.0"
)

# 注册路由
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(items.router, prefix="/items", tags=["items"])


@app.get("/")
def read_root():
    """根路径，返回欢迎信息"""
    return {"message": "Welcome to Test API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
