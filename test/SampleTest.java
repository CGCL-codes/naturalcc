public class SampleTest {
    public static int add(int a, int b) {
        return a + b;
    }

    public static void main(String[] args) {
        int result = add(2, 3);
        if (result == 5) {
            System.out.println("PASS: add(2,3)=5");
        } else {
            System.out.println("FAIL: expected 5, got " + result);
        }
    }
}
