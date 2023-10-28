COMPILER?=nvcc
OPT_FLAGS?=-O3 

BB=256 # 1024
QQ=2 #4

#-arch=compute_35

LDFLAGS?=-lOpenCL

TEST_PBB=test-pbb
TEST_SPMV=test-spmv
TEST_F=test-f
TEST_SPS=test-sps

.PHONY: clean all run

default: compile run

compile: $(TEST_SPS)

$(TEST_SPS): testPBB.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h sps.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_SPS) -D B=$(BB) -D Q=$(QQ) testPBB.cu

$(TEST_F): testSPSFunctions.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h sps.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_F) -D B=4 -D Q=2 testSPSFunctions.cu

all: $(TEST_SPS) 

# 100003565
run: $(TEST_SPS) 
	timeout 15s ./$(TEST_SPS) 265

run-ftest: $(TEST_F)
	timeout 5s ./$(TEST_F) 16

clean:
	rm -f $(TEST_SPS)




