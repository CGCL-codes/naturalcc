package com.example.demo.repository;

/*
 * 项目: project_10_java_spring
 * 测试主题: Java Spring Boot 企业级应用
 * 功能: 用户数据访问层 - Spring Data JPA
 */
import com.example.demo.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByUsername(String username);
    Optional<User> findByEmail(String email);
    boolean existsByUsername(String username);
    boolean existsByEmail(String email);
}
