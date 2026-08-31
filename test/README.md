# StudentManager 文档

## 概述

`StudentManager` 是一个用 Java 编写的简单学生管理器类，位于 `StudentManager.java` 文件中。它提供对学生对象的增、删、查、统计等基本管理功能。该类同时包含一个内部静态类 `Student` 用于表示学生实体。

> **说明**：该项目用于测试 NaturalCC 代码补全和修复功能，部分方法含 TODO 或已知 BUG。

---

## 文件目录

| 文件 | 说明 |
|------|------|
| `StudentManager.java` | 核心文件，包含 `StudentManager` 类和内部类 `Student` |
| `SampleTest.java` | 独立的示例测试类，与 `StudentManager` **无直接调用关系** |
| `README.md` | 本文档 |

> 当前项目中，没有任何其他 Java 文件通过 `import` 或引用方式调用 `StudentManager`。`StudentManager` 是一个**独立的、自包含的**类，其 `main` 方法中包含自测演示逻辑。

---

## StudentManager 功能详解

### 1. 构造函数

```java
public StudentManager()
```

- 初始化一个空的 `ArrayList<Student>` 内部列表。
- 无参数，始终成功。

### 2. 添加学生

```java
public boolean addStudent(String name, int age)
```

| 参数 | 说明 |
|------|------|
| `name` | 学生姓名，不可为 `null` |
| `age`  | 学生年龄，必须为正数 |

- 如果 `name == null` 或 `age <= 0`，返回 `false`。
- 否则，创建新的 `Student` 对象并加入列表，返回 `true`。

### 3. 查找学生

```java
public Student findStudent(String name)
```

- 遍历内部学生列表，按姓名匹配。
- 找到第一个匹配的学生则返回该 `Student` 对象。
- 未找到则返回 `null`。

### 4. 计算平均年龄

```java
public double calculateAverageAge()
```

- 遍历所有学生，累加年龄，计算平均值。
- 如果列表为空，返回 `0.0`（**已修复**——原 BUG 遗漏了空列表检查）。

### 5. 删除学生

```java
public void removeStudent(String name)
```

- 按姓名查找，删除**第一个**匹配的学生。
- **已知 BUG**：当存在同名学生时，只删除第一个，不会处理剩余的重名学生。

### 6. 获取学生数量

```java
public int getStudentCount()
```

- 返回内部列表的大小，即当前管理的学生总数。

---

## 内部类：Student

`Student` 是 `StudentManager` 的一个 **`public static` 内部类**，表示一个学生实体。

### 属性

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `String` | 学生姓名 |
| `age`  | `int`  | 学生年龄 |

### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `Student(String name, int age)` | — | 构造方法 |
| `getName()` | `String` | 获取姓名 |
| `getAge()` | `int` | 获取年龄 |
| `setName(String name)` | `void` | 设置姓名 |
| `setAge(int age)` | `void` | 设置年龄 |
| `toString()` | `String` | 返回格式 `Student{name='...', age=...}` |

---

## 文件调用关系图

```
┌───────────────────────────┐
│     StudentManager.java   │
│                           │
│  ┌─────────────────────┐  │
│  │  StudentManager     │  │  (主类)
│  │  - students: List   │──│── 依赖 java.util.ArrayList
│  │  + StudentManager() │  │       java.util.List
│  │  + addStudent()     │  │
│  │  + findStudent()    │  │
│  │  + calcAvgAge()     │  │
│  │  + removeStudent()  │  │
│  │  + getStudentCount()│  │
│  │  + main()           │  │  (自测入口)
│  └─────────────────────┘  │
│            │               │
│            ▼               │
│  ┌─────────────────────┐  │
│  │  Student (static)   │  │  (内部类)
│  │  - name: String     │  │
│  │  - age: int         │  │
│  │  + getName/setName  │  │
│  │  + getAge/setAge    │  │
│  │  + toString()       │  │
│  └─────────────────────┘  │
└───────────────────────────┘

           │
           │ (无外部引用)
           ▼
┌───────────────────────┐
│    SampleTest.java    │  (独立测试类，与 StudentManager 无关)
└───────────────────────┘

┌───────────────────────┐
│  Java 标准库依赖       │
│  - java.util.ArrayList │
│  - java.util.List      │
└───────────────────────┘
```

### 调用关系说明

1. **`StudentManager` → `Student`**：`StudentManager` 的 `students` 列表持有 `Student` 类型的对象，各业务方法（`addStudent`、`findStudent`、`calculateAverageAge`、`removeStudent`）均通过 `Student` 对象的 `getName()`、`getAge()` 等方法访问其属性。

2. **`StudentManager` → Java 标准库**：
   - `java.util.ArrayList`：作为底层存储结构。
   - `java.util.List`：作为 `students` 字段的声明类型。

3. **外部文件调用**：
   - 当前工作区中没有其他 Java 源文件引用或调用 `StudentManager`。
   - `SampleTest.java` 是一个完全独立的测试类，仅包含一个 `add` 方法，与 `StudentManager` 无任何关系。

4. **自测入口**：`StudentManager` 自身的 `main()` 方法提供了一个简单的演示：创建管理器、添加两名学生（Alice, Bob）、打印总数和平均年龄。

---

## 已知问题与待办

| 位置 | 类型 | 描述 |
|------|------|------|
| `addStudent()` | ~~TODO~~ (已完成) | 方法体已实现 |
| `findStudent()` | ~~TODO~~ (已完成) | 方法体已实现 |
| `calculateAverageAge()` | ~~BUG~~ (已修复) | 空列表检查已添加 |
| `removeStudent()` | **BUG (未修复)** | 未处理重名学生的情况，只删除第一个匹配项 |

---

## 如何运行

```bash
javac StudentManager.java   # 编译
java StudentManager         # 运行
```

运行输出示例：

```
Total students: 2
Average age: 21.0
```
