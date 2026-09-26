/*
  项目: project_05_landing_page
  测试主题: HTML/CSS 响应式落地页设计
  功能: 主交互脚本 - 平滑滚动、移动端菜单、导航栏效果
*/

// 等待 DOM 加载完成
document.addEventListener('DOMContentLoaded', () => {
  initSmoothScroll();
  initMobileMenu();
  initScrollEffect();
  initButtonHandlers();
});

/**
 * 平滑滚动到锚点
 */
function initSmoothScroll() {
  const links = document.querySelectorAll('a[href^="#"]');
  
  links.forEach(link => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      
      // 跳过空锚点或只有 # 的链接
      if (href === '#' || href === '#signup' || href === '#demo') {
        e.preventDefault();
        console.log(`[Demo] 点击了按钮: ${link.textContent}`);
        return;
      }
      
      // 获取目标元素
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        
        // 如果是移动端，关闭菜单
        const navLinks = document.querySelector('.nav-links');
        if (navLinks.classList.contains('nav-active')) {
          navLinks.classList.remove('nav-active');
        }
        
        // 平滑滚动到目标
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
}

/**
 * 移动端菜单切换
 */
function initMobileMenu() {
  const menuToggle = document.querySelector('.mobile-menu-toggle');
  const navLinks = document.querySelector('.nav-links');
  
  if (menuToggle && navLinks) {
    menuToggle.addEventListener('click', () => {
      navLinks.classList.toggle('nav-active');
      
      // 更新按钮的 aria 状态
      const isActive = navLinks.classList.contains('nav-active');
      menuToggle.setAttribute('aria-expanded', isActive);
      
      // 切换图标
      menuToggle.textContent = isActive ? '✕' : '☰';
      
      console.log(`[移动端菜单] ${isActive ? '打开' : '关闭'}`);
    });
    
    // 点击导航链接后自动关闭菜单
    const navLinkItems = navLinks.querySelectorAll('a');
    navLinkItems.forEach(link => {
      link.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
          navLinks.classList.remove('nav-active');
          menuToggle.textContent = '☰';
          menuToggle.setAttribute('aria-expanded', 'false');
        }
      });
    });
  }
}

/**
 * 滚动时导航栏效果
 */
function initScrollEffect() {
  const header = document.querySelector('.site-header');
  let lastScrollY = window.scrollY;
  
  window.addEventListener('scroll', () => {
    const currentScrollY = window.scrollY;
    
    // 滚动超过 50px 时添加阴影
    if (currentScrollY > 50) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
    
    lastScrollY = currentScrollY;
  });
}

/**
 * 按钮点击处理（演示用）
 */
function initButtonHandlers() {
  // 为所有按钮添加演示处理
  const buttons = document.querySelectorAll('.btn-primary, .btn-secondary');
  
  buttons.forEach(button => {
    if (button.tagName === 'BUTTON') {
      button.addEventListener('click', (e) => {
        e.preventDefault();
        const buttonText = button.textContent.trim();
        console.log(`[按钮点击] ${buttonText}`);
        
        // 可以在这里添加实际的功能，如打开模态框、跳转页面等
        alert(`演示功能：${buttonText}`);
      });
    }
  });
  
  // 定价卡片按钮特殊处理
  const pricingButtons = document.querySelectorAll('.pricing-card button');
  pricingButtons.forEach(button => {
    button.addEventListener('click', (e) => {
      e.preventDefault();
      const card = button.closest('.pricing-card');
      const plan = card.querySelector('h3').textContent;
      const price = card.querySelector('.price').textContent;
      
      console.log(`[定价方案] 选择了: ${plan} - ${price}`);
      alert(`您选择了 ${plan}\n价格: ${price}\n\n这是一个演示项目，实际功能需要后端支持。`);
    });
  });
}

// 添加页面可见性变化监听（可选的性能优化）
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    console.log('[页面状态] 页面已隐藏');
  } else {
    console.log('[页面状态] 页面已显示');
  }
});

// 窗口大小改变时关闭移动菜单
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const navLinks = document.querySelector('.nav-links');
    const menuToggle = document.querySelector('.mobile-menu-toggle');
    
    // 如果窗口变大，确保菜单是关闭的
    if (window.innerWidth > 768 && navLinks.classList.contains('nav-active')) {
      navLinks.classList.remove('nav-active');
      menuToggle.textContent = '☰';
      menuToggle.setAttribute('aria-expanded', 'false');
    }
  }, 250);
});

console.log('[TestSaaS] 页面脚本已加载完成');
