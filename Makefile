
ifndef CPPC
	CPPC=g++
endif


CCFLAGS=



LIBS = -lOpenCL -lrt




all: mt_sng_test.o mt_cpu.o 
	$(CPPC) $^   $(CCFLAGS) $(LIBS)    -o mt


mt_cpu.o: mt_cpu.cpp
	$(CPPC)	$^ -c  $(CCFLAGS) $(LIBS)  -o $@
	
mt_sng_test.o: mt_sng_test.cpp
	$(CPPC)	 $^ -c  $(CCFLAGS) $(LIBS) -o $@	

clean:
	rm -f mt
