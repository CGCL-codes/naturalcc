import java.util.ArrayList;
import java.util.List;

/**
 * 一个简单的学生管理器，功能尚未完善。
 * 用于测试 NaturalCC 代码补全和修复功能。
 */
public class StudentManager {
    private List<Student> students;

    /**
     * 构造一个空的学生管理器。
     */
    public StudentManager() {
        this.students = new ArrayList<>();
    }

    /**
     * 添加一名具有给定姓名和年龄的学生。
     *
     * @param name 学生的姓名（不能为 null）
     * @param age  学生的年龄（必须为正数）
     * @return 如果学生添加成功则返回 true，否则返回 false
     */
    // TODO: 完成此方法 - 应添加学生并在成功时返回 true
    public boolean addStudent(String name, int age) {
        if (name == null || age <= 0) {
            return false;
        }
        students.add(new Student(name, age));
        return true;
    }

    /**
     * 根据姓名查找学生。
     *
     * @param name 要搜索的姓名
     * @return 具有给定姓名的 Student 对象，如果未找到则返回 null
     */
    // TODO: 完成此方法 - 应按姓名查找学生
    public Student findStudent(String name) {
        for (Student s : students) {
            if (s.getName().equals(name)) {
                return s;
            }
        }
        return null;
    }

    /**
     * 计算所有学生的平均年龄。
     *
     * @return 平均年龄，如果没有学生则返回 0.0
     */
    // BUG: 应在计算平均年龄前检查列表是否为空
    public double calculateAverageAge() {
        if (students.isEmpty()) {
            return 0.0;
        }
        int total = 0;
        for (Student s : students) {
            total += s.getAge();
        }
        return (double) total / students.size();
    }

    /**
     * 删除具有给定姓名的第一个学生。
     *
     * @param name 要删除的学生姓名
     */
    // BUG: 未处理重名情况
    public void removeStudent(String name) {
        for (int i = 0; i < students.size(); i++) {
            if (students.get(i).getName().equals(name)) {
                students.remove(i);
                break;
            }
        }
    }

    /**
     * 返回当前管理的学生数量。
     *
     * @return 学生数量
     */
    public int getStudentCount() {
        return students.size();
    }

    /**
     * 表示一个具有姓名和年龄的学生。
     */
    public static class Student {
        private String name;
        private int age;

        /**
         * 用给定的姓名和年龄构造一个学生。
         *
         * @param name 学生的姓名
         * @param age  学生的年龄
         */
        public Student(String name, int age) {
            this.name = name;
            this.age = age;
        }

        /**
         * 返回学生的姓名。
         *
         * @return 姓名
         */
        public String getName() { return name; }

        /**
         * 返回学生的年龄。
         *
         * @return 年龄
         */
        public int getAge() { return age; }

        /**
         * 设置学生的姓名。
         *
         * @param name 新姓名
         */
        public void setName(String name) { this.name = name; }

        /**
         * 设置学生的年龄。
         *
         * @param age 新年龄
         */
        public void setAge(int age) { this.age = age; }

        @Override
        public String toString() {
            return "Student{name='" + name + "', age=" + age + "}";
        }
    }

    /**
     * 用于演示的主方法。
     *
     * @param args 命令行参数（未使用）
     */
    public static void main(String[] args) {
        StudentManager manager = new StudentManager();
        manager.addStudent("Alice", 20);
        manager.addStudent("Bob", 22);
        System.out.println("Total students: " + manager.getStudentCount());
        System.out.println("Average age: " + manager.calculateAverageAge());
    }
}
