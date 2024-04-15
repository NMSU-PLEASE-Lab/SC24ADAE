
APPEKGDIR = ..

# on Mac, -Xclang may be needed
CFLAGS = -I${APPEKGDIR} -Wall -fopenmp
LDOBJS = -L${APPEKGDIR} -lappekg -lpthread 

all: binary hello evensum

hello: HelloWorld.o
	gcc -o $@ $< ${LDOBJS} 

binary: BinarySearch.o
	gcc -o $@ $< ${LDOBJS} 

binary_linear: BinaryVsLinear.o
	gcc -o $@ $< ${LDOBJS}

binary_linear_multithread: BinaryVsLinearMultithreading.o
	gcc -o $@ $< ${LDOBJS}

evensum: EvenNumbersSum.o
	gcc -fopenmp -o $@ $< ${LDOBJS}

dynamic_hb: DynamicHBGenerator.o
	gcc -o $@ $< ${LDOBJS}

threadtest: threadtest.c
	gcc -o $@ -fopenmp -Wall threadtest.c

clean:
	rm -f *.o hello binary binary_linear binary_linear_multithread evensum dynamic_hb threadtest
