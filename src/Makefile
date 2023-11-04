COMPILER?=nvcc
OPT_FLAGS?=-O3 -diag-suppress 177
# We suppress warning from the GPU Scan borrowed from assignment 2

BB=256# base: 256
QQ=30# base: 30

#-arch=compute_35

LDFLAGS?=-lOpenCL

TEST_PBB=test-pbb
TEST_SPMV=test-spmv
TEST_F=test-f
TEST_SPS=test-sps
TEST_TABLE=test-table
TEST_HEATMAPS=test-heatmaps

.PHONY: clean all run

default: compile run

compile: $(TEST_SPS)

$(TEST_SPS): testSPS.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h spsKernels.cu.h
	$(COMPILER) $(OPT_FLAGS)  -o $(TEST_SPS) -D B=$(BB) -D Q=$(QQ) -D INCLUDE_QUAD=$(INCLUDE_QUAD) -D PARALLEL_REDUCTION=$(PARALLEL_REDUCTION) testSPS.cu

$(TEST_F): testSPSFunctions.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h spsKernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_F) -D B=$(BB) -D Q=$(QQ) testSPSFunctions.cu

$(TEST_TABLE): testSPStable.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h spsKernels.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_TABLE) -D B=$(BB) -D Q=$(QQ) -D INCLUDE_QUAD=$(INCLUDE_QUAD) testSPStable.cu

all: $(TEST_SPS)

# 100003565 1073741824
run: $(TEST_SPS)
	./$(TEST_SPS) 100003565

run-ftest: $(TEST_F)
	timeout 10s ./$(TEST_F) 400

run-table: $(TEST_TABLE)
	./$(TEST_TABLE) >> results-$(BB)-$(QQ)-int32.csv


run-heatmaps: testSPSheatmaps.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h spsKernels.cu.h
	for B in 32 64 128 256 512 1024; do \
		for Q in 2 4 7 8 10 13 16 20 24 30 32 40; do \
			$(COMPILER) $(OPT_FLAGS) -o $(TEST_HEATMAPS) -D B=$$B -D Q=$$Q testSPSheatmaps.cu; \
			./$(TEST_HEATMAPS) >> heatmaps-b$$B.txt; \
		done \
	done

clean:
	rm -f $(TEST_SPS) $(TEST_F)
