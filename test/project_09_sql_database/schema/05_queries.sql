-- =============================================================================
-- 项目: project_09_sql_database
-- 测试主题: SQL 数据库架构设计 - 常用查询示例
-- 功能: 业务查询示例
-- =============================================================================

-- 1. 查询某用户的所有订单
SELECT * FROM v_order_details
WHERE user_id = 1
ORDER BY created_at DESC;

-- 2. 查询本月销售总额
SELECT
    DATE_TRUNC('day', created_at) AS day,
    SUM(total_amount) AS daily_revenue,
    COUNT(*) AS order_count
FROM orders
WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)
    AND status IN ('paid', 'shipped', 'delivered')
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY day;

-- 3. TOP 10 销量商品
SELECT * FROM v_product_sales_stats
WHERE total_sold > 0
ORDER BY total_sold DESC
LIMIT 10;

-- 4. 高价值用户（消费 > 10000）
SELECT * FROM v_user_spending_stats
WHERE total_spent > 10000
ORDER BY total_spent DESC;

-- 5. 商品库存预警（库存 < 100）
SELECT id, sku, name, stock
FROM products
WHERE stock < 100 AND status = 'on_sale'
ORDER BY stock ASC;

-- 6. 待支付订单（创建超过 30 分钟）
SELECT order_no, user_id, total_amount, created_at,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 60 AS minutes_ago
FROM orders
WHERE status = 'pending'
    AND created_at < CURRENT_TIMESTAMP - INTERVAL '30 minutes'
ORDER BY created_at;

-- 7. 用户最近一次订单
SELECT DISTINCT ON (user_id)
    user_id, order_no, total_amount, status, created_at
FROM orders
ORDER BY user_id, created_at DESC;

-- 8. 分类销售占比
SELECT
    c.name AS category,
    SUM(oi.subtotal) AS revenue,
    ROUND(SUM(oi.subtotal) * 100.0 / SUM(SUM(oi.subtotal)) OVER (), 2) AS percentage
FROM categories c
JOIN products p ON p.category_id = c.id
JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
WHERE o.status IN ('paid', 'shipped', 'delivered')
GROUP BY c.name
ORDER BY revenue DESC;
