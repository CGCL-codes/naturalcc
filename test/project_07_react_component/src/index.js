/*
  项目: project_07_react_component
  测试主题: React 组件开发与状态管理
  功能: 应用入口
*/
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
