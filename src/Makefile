COMPILER?=nvcc
OPT_FLAGS?=-O3 

BB=1024
QQ=4

#-arch=compute_35

LDFLAGS?=-lOpenCL

TEST_SPS=test-sps

.PHONY: clean all run

default: compile run

compile: $(TEST_SPS)

$(TEST_SPS): testPBB.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_SPS) -D B=$(BB) -D Q=$(QQ) testPBB.cu


all: $(TEST_SPS) 


run: $(TEST_SPS) 
	./$(TEST_SPS) 100003565


clean:
	rm -f $(TEST_SPS)




