/*
  项目: project_07_react_component
  测试主题: React 组件开发与状态管理
  功能: Todo 表单组件 - 输入框和提交按钮
*/
import React, { useState } from 'react';

function TodoForm({ onAdd }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onAdd(text);
    setText('');
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="输入新任务..."
        style={styles.input}
      />
      <button type="submit" style={styles.button}>添加</button>
    </form>
  );
}

const styles = {
  form: { display: 'flex', gap: 8, marginBottom: 20 },
  input: {
    flex: 1,
    padding: '10px 14px',
    fontSize: 16,
    border: '1px solid #d1d5db',
    borderRadius: 6,
  },
  button: {
    padding: '10px 20px',
    background: '#4f46e5',
    color: '#fff',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
  },
};

export default TodoForm;
