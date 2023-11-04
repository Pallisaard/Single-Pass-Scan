COMPILER?=nvcc
OPT_FLAGS?=-O3 -diag-suppress 177

BB=1024# base: 1024
QQ=10# base: 10
INCLUDE_QUAD=0 # 0 = int32, 1 = Quad<int32>
PARALLEL_REDUCTION=1 # 0 = no parallel reduction, 1 = parallel reduction

#-arch=compute_35
# NOTE COMPARE WITH VOLATILE OFF AND REMEMBER TO
# REPLACE WITH __threadfence()

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
	./$(TEST_SPS) 100

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
