-- =============================================================================
-- 项目: project_09_sql_database
-- 测试主题: SQL 数据库架构设计 - 测试数据
-- 功能: 插入测试数据用于开发与测试
-- =============================================================================

-- 用户测试数据
INSERT INTO users (username, email, password_hash, phone) VALUES
    ('alice', 'alice@example.com', '$2b$10$abcdefghijklmnopqrstuv', '13800138001'),
    ('bob', 'bob@example.com', '$2b$10$abcdefghijklmnopqrstuv', '13800138002'),
    ('charlie', 'charlie@example.com', '$2b$10$abcdefghijklmnopqrstuv', '13800138003');

INSERT INTO user_profiles (user_id, full_name, gender) VALUES
    (1, 'Alice Wang', 'female'),
    (2, 'Bob Li', 'male'),
    (3, 'Charlie Zhang', 'male');

-- 分类测试数据
INSERT INTO categories (name, slug, sort_order) VALUES
    ('电子产品', 'electronics', 1),
    ('图书', 'books', 2),
    ('服装', 'clothing', 3);

INSERT INTO categories (name, parent_id, slug, sort_order) VALUES
    ('手机', 1, 'phones', 1),
    ('笔记本', 1, 'laptops', 2),
    ('小说', 2, 'fiction', 1);

-- 商品测试数据
INSERT INTO products (sku, name, description, price, stock, category_id) VALUES
    ('PHN-001', 'iPhone 15 Pro', '最新款苹果手机', 8999.00, 100, 4),
    ('PHN-002', '小米 14', '高性价比安卓手机', 3999.00, 200, 4),
    ('LAP-001', 'MacBook Pro 14', '苹果笔记本电脑', 14999.00, 50, 5),
    ('LAP-002', 'ThinkPad X1', '商务笔记本', 12999.00, 80, 5),
    ('BK-001', '三体', '刘慈欣科幻小说', 39.90, 500, 6),
    ('BK-002', '活着', '余华经典小说', 28.00, 300, 6);

-- 订单测试数据
INSERT INTO orders (order_no, user_id, status, total_amount, shipping_address) VALUES
    ('ORD20260907001', 1, 'paid', 9038.90, '{"city": "北京", "address": "中关村大街1号"}'),
    ('ORD20260907002', 2, 'shipped', 14999.00, '{"city": "上海", "address": "陆家嘴环路100号"}'),
    ('ORD20260907003', 1, 'pending', 39.90, '{"city": "北京", "address": "中关村大街1号"}');

INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal) VALUES
    (1, 1, 1, 8999.00, 8999.00),
    (1, 5, 1, 39.90, 39.90),
    (2, 3, 1, 14999.00, 14999.00),
    (3, 5, 1, 39.90, 39.90);

INSERT INTO payments (order_id, payment_method, transaction_id, amount, status, paid_at) VALUES
    (1, 'alipay', 'TXN20260907001', 9038.90, 'success', CURRENT_TIMESTAMP),
    (2, 'wechat', 'TXN20260907002', 14999.00, 'success', CURRENT_TIMESTAMP);
