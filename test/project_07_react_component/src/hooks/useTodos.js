/*
  项目: project_07_react_component
  测试主题: React 组件开发与状态管理
  功能: 自定义 Hook - 管理 Todo 状态
*/
import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'test_todos';

export function useTodos() {
  const [todos, setTodos] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  // 持久化到 localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(todos));
  }, [todos]);

  const addTodo = useCallback((text) => {
    if (!text.trim()) return;
    setTodos(prev => [
      ...prev,
      {
        id: Date.now() + Math.random(),
        text: text.trim(),
        completed: false,
        createdAt: new Date().toISOString(),
      },
    ]);
  }, []);

  const toggleTodo = useCallback((id) => {
    setTodos(prev => prev.map(t =>
      t.id === id ? { ...t, completed: !t.completed } : t
    ));
  }, []);

  const deleteTodo = useCallback((id) => {
    setTodos(prev => prev.filter(t => t.id !== id));
  }, []);

  const clearCompleted = useCallback(() => {
    setTodos(prev => prev.filter(t => !t.completed));
  }, []);

  return { todos, addTodo, toggleTodo, deleteTodo, clearCompleted };
}
