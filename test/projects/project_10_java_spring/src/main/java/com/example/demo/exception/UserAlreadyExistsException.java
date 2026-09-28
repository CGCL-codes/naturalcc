package com.example.demo.exception;

/*
 * 项目: project_10_java_spring
 * 测试主题: Java Spring Boot 企业级应用
 * 功能: 用户已存在异常
 */
public class UserAlreadyExistsException extends RuntimeException {
    public UserAlreadyExistsException(String message) {
        super(message);
    }
}
