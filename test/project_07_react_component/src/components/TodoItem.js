/*
  项目: project_07_react_component
  测试主题: React 组件开发与状态管理
  功能: 单个 Todo 项组件
*/
import React from 'react';

function TodoItem({ todo, onToggle, onDelete }) {
  return (
    <li style={{
      ...styles.item,
      ...(todo.completed ? styles.completed : {}),
    }}>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={() => onToggle(todo.id)}
      />
      <span style={styles.text}>{todo.text}</span>
      <button onClick={() => onDelete(todo.id)} style={styles.delete}>
        ✕
      </button>
    </li>
  );
}

const styles = {
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '12px 16px',
    border: '1px solid #e5e7eb',
    borderRadius: 6,
    marginBottom: 8,
  },
  completed: { opacity: 0.6, background: '#f9fafb' },
  text: { flex: 1 },
  delete: {
    border: 'none',
    background: 'transparent',
    cursor: 'pointer',
    color: '#ef4444',
    fontSize: 16,
  },
};

export default TodoItem;
