import java.util.Scanner;
import java.util.Arrays;
public class Fibonaci {
	public static long[] fibonaci(int n) {
		if(n <= 1) {
			long[] answer = {n, 0};
			return answer;
		}
		long[] temp  = fibonaci(n-1) ;
		long[] answer = {temp[0] + temp[1], temp[0]};
		return answer;
	}
	public static void main(String[] agrs) {
		Scanner input = new Scanner(System.in);
		int n = input.nextInt();
		System.out.println(Arrays.toString(fibonaci(n)));
	}
}