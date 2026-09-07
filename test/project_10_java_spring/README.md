# 测试主题：Java Spring Boot 企业级应用

这是一个用于测试 coding agent Java 企业级开发能力的项目。

## 测试目标
- 测试 agent 是否能正确创建 Spring Boot 项目结构
- 测试 agent 是否能正确使用 Spring Data JPA / MyBatis
- 测试 agent 是否能正确实现 REST 控制器和服务层
- 测试 agent 是否能正确处理异常、事务、安全（Spring Security）

## 项目结构
```
project_10_java_spring/
├── README.md
├── pom.xml                          # Maven 配置
├── src/main/java/com/example/demo/
│   ├── DemoApplication.java        # 主类
│   ├── controller/                  # 控制器层
│   ├── service/                     # 服务层
│   ├── repository/                  # 数据访问层
│   ├── entity/                      # 实体类
│   ├── dto/                         # 数据传输对象
│   ├── config/                      # 配置类
│   └── exception/                   # 异常处理
└── src/main/resources/
    └── application.yml              # 应用配置
```
