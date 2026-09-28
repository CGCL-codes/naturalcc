/*
  项目: project_02_js_calculator
  测试主题: JavaScript 前端交互 - 计算器应用
  功能: 计算器核心逻辑 - 处理按钮点击、键盘输入、表达式计算
*/

class Calculator {
  constructor(displayElement) {
    this.display = displayElement;
    this.currentValue = '0';
    this.previousValue = null;
    this.operator = null;
    this.shouldResetDisplay = false;
    this.init();
  }

  init() {
    // 绑定按钮事件
    document.querySelectorAll('.btn').forEach(btn => {
      btn.addEventListener('click', () => this.handleButton(btn.textContent));
    });

    // 绑定键盘事件
    document.addEventListener('keydown', (e) => this.handleKeyboard(e));
  }

  handleButton(label) {
    if (/[0-9]/.test(label)) {
      this.inputNumber(label);
    } else if (label === '.') {
      this.inputDecimal();
    } else if (['+', '-', '*', '/'].includes(label)) {
      this.inputOperator(label);
    } else if (label === '=') {
      this.calculate();
    } else if (label === 'C') {
      this.clear();
    } else if (label === '←') {
      this.backspace();
    }
    this.updateDisplay();
  }

  handleKeyboard(e) {
    const key = e.key;
    if (/[0-9]/.test(key)) {
      this.handleButton(key);
    } else if (['+', '-', '*', '/'].includes(key)) {
      this.handleButton(key);
    } else if (key === 'Enter' || key === '=') {
      this.handleButton('=');
    } else if (key === 'Backspace') {
      this.handleButton('←');
    } else if (key === 'Escape') {
      this.handleButton('C');
    } else if (key === '.') {
      this.handleButton('.');
    }
  }

  inputNumber(num) {
    if (this.shouldResetDisplay) {
      this.currentValue = num;
      this.shouldResetDisplay = false;
    } else {
      this.currentValue = this.currentValue === '0' ? num : this.currentValue + num;
    }
  }

  inputDecimal() {
    if (this.shouldResetDisplay) {
      this.currentValue = '0.';
      this.shouldResetDisplay = false;
    } else if (!this.currentValue.includes('.')) {
      this.currentValue += '.';
    }
  }

  inputOperator(op) {
    if (this.operator && !this.shouldResetDisplay) {
      this.calculate();
    }
    this.previousValue = this.currentValue;
    this.operator = op;
    this.shouldResetDisplay = true;
  }

  calculate() {
    if (!this.operator || this.previousValue === null) return;
    const prev = parseFloat(this.previousValue);
    const curr = parseFloat(this.currentValue);
    let result;
    switch (this.operator) {
      case '+': result = prev + curr; break;
      case '-': result = prev - curr; break;
      case '*': result = prev * curr; break;
      case '/':
        if (curr === 0) {
          this.currentValue = '错误';
          this.operator = null;
          this.previousValue = null;
          this.shouldResetDisplay = true;
          return;
        }
        result = prev / curr;
        break;
    }
    this.currentValue = String(parseFloat(result.toFixed(8)));
    this.operator = null;
    this.previousValue = null;
    this.shouldResetDisplay = true;
  }

  clear() {
    this.currentValue = '0';
    this.previousValue = null;
    this.operator = null;
    this.shouldResetDisplay = false;
  }

  backspace() {
    if (this.shouldResetDisplay) return;
    this.currentValue = this.currentValue.length > 1
      ? this.currentValue.slice(0, -1)
      : '0';
  }

  updateDisplay() {
    this.display.value = this.currentValue;
  }
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
  const display = document.getElementById('display');
  new Calculator(display);
});
