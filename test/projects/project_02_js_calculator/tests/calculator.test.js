/*
  项目: project_02_js_calculator
  测试主题: JavaScript 前端交互 - 计算器应用
  功能: 计算器单元测试（使用 Jest 风格）
*/

// 简单的测试框架
const tests = [];
function test(name, fn) {
  tests.push({ name, fn });
}

function assertEqual(actual, expected) {
  if (actual !== expected) {
    throw new Error(`Expected ${expected}, got ${actual}`);
  }
}

test('输入数字', () => {
  const calc = new Calculator({ value: '' });
  calc.inputNumber('5');
  assertEqual(calc.currentValue, '5');
});

test('加法运算', () => {
  const calc = new Calculator({ value: '' });
  calc.inputNumber('2');
  calc.inputOperator('+');
  calc.inputNumber('3');
  calc.calculate();
  assertEqual(calc.currentValue, '5');
});

test('除零错误', () => {
  const calc = new Calculator({ value: '' });
  calc.inputNumber('5');
  calc.inputOperator('/');
  calc.inputNumber('0');
  calc.calculate();
  assertEqual(calc.currentValue, '错误');
});

test('清除功能', () => {
  const calc = new Calculator({ value: '' });
  calc.inputNumber('9');
  calc.clear();
  assertEqual(calc.currentValue, '0');
});

// 导出供测试运行器使用
if (typeof module !== 'undefined') {
  module.exports = { tests };
}
