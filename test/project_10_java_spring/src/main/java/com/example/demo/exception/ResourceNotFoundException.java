package com.example.demo.exception;

/*
 * 项目: project_10_java_spring
 * 测试主题: Java Spring Boot 企业级应用
 * 功能: 资源未找到异常
 */
public class ResourceNotFoundException extends RuntimeException {
    public ResourceNotFoundException(String message) {
        super(message);
    }
}
