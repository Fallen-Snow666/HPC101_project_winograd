# Makefile for Winograd Project with CUTLASS

# --- 编译器和目标 ---
NVCC = nvcc
TARGET = winograd
CUTLASS_DIR = ./cutlass
SOURCES = main.cu naive_conv.cu winograd_conv.cu
NCCL_INCLUDE_PATH = /pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/nvhpc-25.1-gfpvhsdurdxu5qqwgkxsn6m76eohxn25/Linux_x86_64/25.1/comm_libs/12.6/nccl/include
NCCL_LIB_PATH = /pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/nvhpc-25.1-gfpvhsdurdxu5qqwgkxsn6m76eohxn25/Linux_x86_64/25.1/comm_libs/12.6/nccl/lib

# --- 编译和链接标志 ---
# -arch=sm_70 针对 Volta 架构 (如 V100) 编译
# -I$(CUTLASS_DIR)/include 添加 CUTLASS 头文件搜索路径
# -lnccl 链接 NCCL 库
# -lpthread 链接线程库 (用于 std::thread)
NVCCFLAGS = -O3 -std=c++17 -arch=sm_70 -I$(CUTLASS_DIR)/include -I$(NCCL_INCLUDE_PATH)
LDFLAGS = -L$(NCCL_LIB_PATH) -lnccl -lpthread

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "===> Compiling and Linking all sources..."
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)
	@echo "===> Build finished successfully: $(TARGET)"

clean:
	@echo "===> Cleaning build files..."
	rm -f $(TARGET)
	@echo "===> Clean complete."

.PHONY: all clean