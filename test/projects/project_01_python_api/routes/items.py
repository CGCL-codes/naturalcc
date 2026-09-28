"""
项目: project_01_python_api
测试主题: Python FastAPI REST API 设计
功能: 商品相关路由 - 包含创建、查询、更新、删除商品
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from database import get_db
from models import Item, User

router = APIRouter()


class ItemCreate(BaseModel):
    """创建商品的请求模型"""
    name: str
    description: Optional[str] = None
    price: float
    owner_id: int


class ItemResponse(BaseModel):
    """商品响应模型"""
    id: int
    name: str
    description: Optional[str]
    price: float
    owner_id: int

    class Config:
        from_attributes = True


@router.post("/", response_model=ItemResponse, status_code=201)
def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    """创建新商品"""
    owner = db.query(User).filter(User.id == item.owner_id).first()
    if not owner:
        raise HTTPException(status_code=404, detail="Owner not found")
    new_item = Item(
        name=item.name,
        description=item.description,
        price=item.price,
        owner_id=item.owner_id
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item


@router.get("/", response_model=List[ItemResponse])
def list_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """获取商品列表"""
    items = db.query(Item).offset(skip).limit(limit).all()
    return items


@router.get("/{item_id}", response_model=ItemResponse)
def get_item(item_id: int, db: Session = Depends(get_db)):
    """根据 ID 获取商品"""
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
