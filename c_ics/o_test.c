#include<stdio.h>
int main(){
	unsigned int a = 2147483648;
	printf("a = %d\n", a);
	printf("a = %u\n", a);
    //输出格式决定显示的值，机器数看a本身。

	unsigned long long b = 2 * 2147483648 * 2147483648 - 1;
	printf("%lld\n", b);
	printf("%llu\n", b);
    printf("%d\n", -9223372036854775808 < 9223372036854775807 );
    printf("%lld\n", 9223372036854775808 );
    printf("%llu\n", - 9223372036854775808 );
    printf("sizeof(int) => %d\n", sizeof(int));
    printf("sizeof(long int) => %d\n", sizeof(long int));
	return 0;
}