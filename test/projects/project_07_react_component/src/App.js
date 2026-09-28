/*
  项目: project_07_react_component
  测试主题: React 组件开发与状态管理
  功能: 主应用组件
*/
import React from 'react';
import { useTodos } from './hooks/useTodos';
import TodoList from './components/TodoList';
import TodoForm from './components/TodoForm';

function App() {
  const { todos, addTodo, toggleTodo, deleteTodo, clearCompleted } = useTodos();

  const remainingCount = todos.filter(t => !t.completed).length;

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1>📝 测试 Todo 应用</h1>
        <p>剩余任务: {remainingCount}</p>
      </header>

      <TodoForm onAdd={addTodo} />

      <TodoList
        todos={todos}
        onToggle={toggleTodo}
        onDelete={deleteTodo}
      />

      {todos.some(t => t.completed) && (
        <button onClick={clearCompleted} style={styles.clearBtn}>
          清除已完成
        </button>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: 600,
    margin: '40px auto',
    padding: '0 20px',
    fontFamily: 'system-ui, sans-serif',
  },
  header: { textAlign: 'center', marginBottom: 24 },
  clearBtn: {
    marginTop: 16,
    padding: '8px 16px',
    background: '#fee2e2',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
    width: '100%',
  },
};

export default App;
