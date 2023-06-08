public class TestJavaClass {
    
    public static int addNumbers(int a, int b) {
        return a + b;
    }
    
    public static void main(String[] args) {
        int result = addNumbers(Integer.parseInt(args[0]), Integer.parseInt(args[1]));
        System.out.println(result);
    }
}
