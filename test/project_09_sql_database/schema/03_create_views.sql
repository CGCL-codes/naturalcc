-- =============================================================================
-- 项目: project_09_sql_database
-- 测试主题: SQL 数据库架构设计 - 视图定义
-- 功能: 创建常用查询视图，简化业务查询
-- =============================================================================

-- 订单详情视图
CREATE VIEW v_order_details AS
SELECT
    o.id AS order_id,
    o.order_no,
    o.user_id,
    u.username,
    u.email,
    o.status,
    o.total_amount,
    o.created_at,
    o.shipping_address,
    COUNT(oi.id) AS item_count,
    SUM(oi.quantity) AS total_quantity
FROM orders o
JOIN users u ON u.id = o.user_id
LEFT JOIN order_items oi ON oi.order_id = o.id
GROUP BY o.id, u.username, u.email;

COMMENT ON VIEW v_order_details IS '订单完整详情视图，包含用户和商品统计';

-- 商品销售统计视图
CREATE VIEW v_product_sales_stats AS
SELECT
    p.id AS product_id,
    p.sku,
    p.name,
    c.name AS category_name,
    COALESCE(SUM(oi.quantity), 0) AS total_sold,
    COALESCE(SUM(oi.subtotal), 0) AS total_revenue,
    COUNT(DISTINCT oi.order_id) AS order_count
FROM products p
LEFT JOIN categories c ON c.id = p.category_id
LEFT JOIN order_items oi ON oi.product_id = p.id
LEFT JOIN orders o ON o.id = oi.order_id
    AND o.status IN ('paid', 'shipped', 'delivered')
GROUP BY p.id, p.sku, p.name, c.name;

COMMENT ON VIEW v_product_sales_stats IS '商品销售统计视图';

-- 用户消费统计视图
CREATE VIEW v_user_spending_stats AS
SELECT
    u.id AS user_id,
    u.username,
    u.email,
    COUNT(DISTINCT o.id) AS order_count,
    COALESCE(SUM(o.total_amount), 0) AS total_spent,
    COALESCE(MAX(o.total_amount), 0) AS max_order_amount,
    MAX(o.created_at) AS last_order_at
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
    AND o.status NOT IN ('cancelled', 'refunded')
GROUP BY u.id, u.username, u.email;

COMMENT ON VIEW v_user_spending_stats IS '用户消费统计视图';
