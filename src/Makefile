COMPILER?=nvcc
OPT_FLAGS?=-O3 

BB=256 # base: 1024
QQ=40 # base: 10

#-arch=compute_35
# NOTE COMPARE WITH VOLATILE OFF AND REMEMBER TO
# REPLACE WITH __threadfence()

LDFLAGS?=-lOpenCL

TEST_PBB=test-pbb
TEST_SPMV=test-spmv
TEST_F=test-f
TEST_SPS=test-sps
TEST_TABLE=test-table

.PHONY: clean all run

default: compile run

compile: $(TEST_SPS)

$(TEST_SPS): testSPS.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h sps.cu.h
	$(COMPILER) $(OPT_FLAGS)  -o $(TEST_SPS) -D B=$(BB) -D Q=$(QQ) testSPS.cu

$(TEST_F): testSPSFunctions.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h sps.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_F) -D B=$(BB) -D Q=$(QQ) testSPSFunctions.cu

$(TEST_TABLE): testSPStable.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h sps.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_TABLE) -D B=$(BB) -D Q=$(QQ) testSPStable.cu

all: $(TEST_SPS)

# 100003565 1073741824
run: $(TEST_SPS)
	./$(TEST_SPS) 100003565

run-ftest: $(TEST_F)
	timeout 10s ./$(TEST_F) 400

run-table: $(TEST_TABLE)
	./$(TEST_TABLE)

clean:
	rm -f $(TEST_SPS)




