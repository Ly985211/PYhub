#include<stdio.h>
#include <stdint.h>

typedef union {
	struct
	{
		uint32_t fraction : 23;
		uint32_t exponent : 8;
		uint32_t sign : 1;
	};
	float fval;
	uint32_t val;
} FLOAT;

int main(){
    FLOAT a;
    a.val = 0x7F7FFFF0;
    printf("a = %x\n", a.val);
    printf("sign = %x\n", a.sign);
    printf("fraction = %x\n", a.fraction);
	/*a = 7f7ffff0
	sign = 0
	fraction = 7ffff0*/
	a.sign = 0x0BBBBBBB;
	printf("a = %x\n", a.val);
    printf("sign = %x\n", a.sign);
    printf("fraction = %x\n", a.fraction);
	/*a = ff7ffff0
	sign = 1*/
}