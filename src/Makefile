COMPILER?=nvcc
OPT_FLAGS?=-O3 

#-arch=compute_35

LDFLAGS?=-lOpenCL

TEST_PBB=test-pbb
TEST_SPMV=test-spmv
TEST_F=test-f

.PHONY: clean all run

default: compile run

compile: $(TEST_PBB)

$(TEST_PBB): testPBB.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_PBB) testPBB.cu

$(TEST_F): testSPSFunctions.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h sps.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_F) testSPSFunctions.cu

all: $(TEST_PBB) 


run: $(TEST_PBB) 
	./$(TEST_PBB) 100003565 256

run-pbb: $(TEST_PBB)
	./$(TEST_PBB) 100003565 256

run-ftest: $(TEST_F)
	./$(TEST_F) 16 4

clean:
	rm -f $(TEST_PBB)




