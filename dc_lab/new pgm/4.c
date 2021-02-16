#include <stdio.h>
#include <omp.h>
#include <math.h>

void main(){
    omp_set_num_threads(8);
    int n;
    printf("Enter num");
    scanf("%d",&n);
    long fact=1,fact1=1;
    int i=1;
    #pragma omp parallel for reduction(*:fact)
    for (i = 1; i <=n; i++)
    {
        fact=fact*i;
    }
    printf("Factorial of %d when first private is not used %ld \n",n,fact);

    #pragma omp parallel for firstprivate(fact1) 
    for (i = 1; i <=n; i++)
    {
        fact1=fact1*i;   
    }
    printf("Factorial of %d when first private is used is %ld \n",n,fact1);
}
//gcc -fopenmp programname.c -lm -lgd
//mpicc 8.c -o 88.c
//mpirun -np 5 ./88.c
