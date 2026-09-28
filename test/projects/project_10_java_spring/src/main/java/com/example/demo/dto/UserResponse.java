package com.example.demo.dto;

/*
 * 项目: project_10_java_spring
 * 测试主题: Java Spring Boot 企业级应用
 * 功能: 用户响应 DTO
 */
import com.example.demo.entity.User;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
public class UserResponse {
    private Long id;
    private String username;
    private String email;
    private LocalDateTime createdAt;

    public static UserResponse fromEntity(User user) {
        return new UserResponse(
            user.getId(),
            user.getUsername(),
            user.getEmail(),
            user.getCreatedAt()
        );
    }
}
