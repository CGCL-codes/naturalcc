-- =============================================================================
-- 项目: project_09_sql_database
-- 测试主题: SQL 数据库架构设计 - 索引优化
-- 功能: 为常用查询字段创建索引
-- =============================================================================

-- 用户表索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status) WHERE status != 'active';
CREATE INDEX idx_users_created_at ON users(created_at DESC);

-- 商品表索引
CREATE INDEX idx_products_category ON products(category_id) WHERE status = 'on_sale';
CREATE INDEX idx_products_status ON products(status);
CREATE INDEX idx_products_price ON products(price) WHERE status = 'on_sale';
CREATE INDEX idx_products_name_trgm ON products USING gin (name gin_trgm_ops);

-- 订单表索引
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status) WHERE status IN ('pending', 'paid');
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
CREATE INDEX idx_orders_user_created ON orders(user_id, created_at DESC);

-- 订单明细索引
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- 支付表索引
CREATE INDEX idx_payments_order_id ON payments(order_id);
CREATE INDEX idx_payments_transaction_id ON payments(transaction_id);
CREATE INDEX idx_payments_status ON payments(status) WHERE status = 'pending';

-- 启用 pg_trgm 扩展以支持三元组索引
CREATE EXTENSION IF NOT EXISTS pg_trgm;
