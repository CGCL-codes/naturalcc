/*
  项目: project_07_react_component
  测试主题: React 组件开发与状态管理
  功能: Todo 列表组件
*/
import React from 'react';
import TodoItem from './TodoItem';

function TodoList({ todos, onToggle, onDelete }) {
  if (todos.length === 0) {
    return <p style={styles.empty}>暂无任务，添加一个吧！</p>;
  }

  return (
    <ul style={styles.list}>
      {todos.map(todo => (
        <TodoItem
          key={todo.id}
          todo={todo}
          onToggle={onToggle}
          onDelete={onDelete}
        />
      ))}
    </ul>
  );
}

const styles = {
  list: { listStyle: 'none', padding: 0 },
  empty: { textAlign: 'center', color: '#999', padding: 20 },
};

export default TodoList;
