#include<stdio.h>
int main(){
    // printf("%d\n", 2 * 2147483647);
    // printf("%s\n", 9223372036854775807);
    printf("sizeof 2147483648: %d\n", sizeof(2147483648));
    printf("sizeof long int: %d\n", sizeof(long int));
    // -> C90, long int, 8 Bytes.
    // sizeof long int: 8, which is 4 on pa_nju.

    // printf("%d\n", sizeof(9223372036854775808));
    
    // unsigned int a = 2147483649;
    // a = -a;
    // printf("%u\n", a);

    // printf("%d\n", -9223372036854775808 < 9223372036854775807 );
    // printf("%d\n", 2 * 2147483648 * 2147483648 < 9223372036854775807 );
  
    return 0;
}