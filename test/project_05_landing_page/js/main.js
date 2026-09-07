/*
  项目: project_05_landing_page
  测试主题: HTML/CSS 响应式落地页设计
  功能: 落地页交互脚本 - 平滑滚动、按钮点击、移动端菜单
*/

document.addEventListener('DOMContentLoaded', () => {
  // 平滑滚动
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // 按钮点击反馈
  document.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', (e) => {
      console.log('按钮被点击:', btn.textContent);
      btn.style.transform = 'scale(0.95)';
      setTimeout(() => { btn.style.transform = ''; }, 150);
    });
  });

  // 滚动时导航阴影
  const header = document.querySelector('.site-header');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 10) {
      header.style.boxShadow = '0 2px 10px rgba(0,0,0,0.05)';
    } else {
      header.style.boxShadow = 'none';
    }
  });
});
