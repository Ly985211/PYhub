#include<stdio.h>
int main(){
    // printf("%d\n", 2 * 2147483647);
    // printf("%s\n", 9223372036854775807);
    printf("%d\n", sizeof(2147483648));
    // -> C90, long int, 8 Bytes.
    // printf("%d\n", sizeof(9223372036854775808));
    
    // unsigned int a = 2147483649;
    // a = -a;
    // printf("%u\n", a);

    // printf("%d\n", -9223372036854775808 < 9223372036854775807 );
    // printf("%d\n", 2 * 2147483648 * 2147483648 < 9223372036854775807 );
  
    return 0;
}