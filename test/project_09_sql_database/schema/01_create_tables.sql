-- =============================================================================
-- 项目: project_09_sql_database
-- 测试主题: SQL 数据库架构设计 - 电商系统数据库
-- 功能: 创建电商系统的核心表（用户、商品、订单、支付等）
-- =============================================================================

-- 启用扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 用户表
CREATE TABLE users (
    id              BIGSERIAL PRIMARY KEY,
    uuid            UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    username        VARCHAR(50) UNIQUE NOT NULL,
    email           VARCHAR(100) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    phone           VARCHAR(20),
    status          VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'banned')),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE users IS '用户主表';
COMMENT ON COLUMN users.status IS '用户状态：active/inactive/banned';

-- 用户档案表（一对一）
CREATE TABLE user_profiles (
    user_id         BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    full_name       VARCHAR(100),
    avatar_url      VARCHAR(500),
    bio             TEXT,
    birthday        DATE,
    gender          VARCHAR(10) CHECK (gender IN ('male', 'female', 'other')),
    address         JSONB
);

-- 商品分类表
CREATE TABLE categories (
    id              BIGSERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    parent_id       BIGINT REFERENCES categories(id) ON DELETE SET NULL,
    slug            VARCHAR(100) UNIQUE NOT NULL,
    sort_order      INTEGER DEFAULT 0,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 商品表
CREATE TABLE products (
    id              BIGSERIAL PRIMARY KEY,
    sku             VARCHAR(50) UNIQUE NOT NULL,
    name            VARCHAR(200) NOT NULL,
    description     TEXT,
    price           DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    stock           INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0),
    category_id     BIGINT REFERENCES categories(id) ON DELETE SET NULL,
    status          VARCHAR(20) DEFAULT 'on_sale' CHECK (status IN ('on_sale', 'off_sale', 'deleted')),
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE products IS '商品主表';

-- 订单表
CREATE TABLE orders (
    id              BIGSERIAL PRIMARY KEY,
    order_no        VARCHAR(50) UNIQUE NOT NULL,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    status          VARCHAR(30) DEFAULT 'pending'
                    CHECK (status IN ('pending', 'paid', 'shipped', 'delivered', 'cancelled', 'refunded')),
    total_amount    DECIMAL(12, 2) NOT NULL CHECK (total_amount >= 0),
    paid_at         TIMESTAMP WITH TIME ZONE,
    shipped_at      TIMESTAMP WITH TIME ZONE,
    delivered_at    TIMESTAMP WITH TIME ZONE,
    shipping_address JSONB NOT NULL,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 订单明细表
CREATE TABLE order_items (
    id              BIGSERIAL PRIMARY KEY,
    order_id        BIGINT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id      BIGINT NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    unit_price      DECIMAL(10, 2) NOT NULL CHECK (unit_price >= 0),
    subtotal        DECIMAL(12, 2) NOT NULL CHECK (subtotal >= 0),
    UNIQUE (order_id, product_id)
);

-- 支付记录表
CREATE TABLE payments (
    id              BIGSERIAL PRIMARY KEY,
    order_id        BIGINT NOT NULL REFERENCES orders(id) ON DELETE RESTRICT,
    payment_method  VARCHAR(30) NOT NULL CHECK (payment_method IN ('alipay', 'wechat', 'credit_card', 'bank_transfer')),
    transaction_id  VARCHAR(100) UNIQUE,
    amount          DECIMAL(12, 2) NOT NULL CHECK (amount > 0),
    status          VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'success', 'failed', 'refunded')),
    paid_at         TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
