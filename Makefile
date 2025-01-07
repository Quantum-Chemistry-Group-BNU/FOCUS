
machine = wuhan #dell2 #scv7260 #scy0799 #DCU_419 #mac #dell #lenovo

DEBUG = no # yes
USE_GCC = yes
USE_MPI = yes
USE_OPENMP = yes
USE_ILP64 = yes
USE_GPU = yes
USE_NCCL = yes
USE_BLAS = yes
# compression
USE_LZ4 = no
USE_ZSTD = no
# exec
INSTALL_CI = yes
INSTALL_CTNS = yes
INSTALL_VMC = yes

# set library
ifeq ($(strip $(machine)), lenovo)
   MATHLIB = /opt/intel/oneapi/mkl/2022.0.2/lib/intel64
   BOOST = /home/lx/software/boost/install_1_79_0
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
else ifeq ($(strip $(machine)), dell)
   MATHLIB = /opt/intel/oneapi/mkl/2022.0.2/lib/intel64
   BOOST = /home/dell/lzd/boost/install
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
else ifeq ($(strip $(machine)), dell2)
   MATHLIB = /home/dell/intel/oneapi/mkl/2022.0.2
   BOOST = /home/dell/users/lzd/boost/install
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
else ifeq ($(strip $(machine)), jiageng)
   MATHLIB = ./mkl2022 #/public/software/intel/oneapi2021/mkl/latest #/public/software/anaconda/anaconda3-2022.5/lib
   BOOST = /public/home/bnulizdtest/boost/install-gcc
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
   #FLAGS += -no-multibyte-chars
else ifeq ($(strip $(machine)), scy0799)
   MATHLIB =/data/apps/OneApi/2022.1/oneapi/mkl/latest/lib/intel64/
   BOOST =/data01/home/scy0799/run/xiangchunyang/project/boost_1_80_0_install
   LFLAGS = -L${BOOST}/lib -lboost_chrono-mt-x64 -lboost_timer-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
else ifeq ($(strip $(machine)), wuhan)
   MATHLIB =/home/share/l6eub2ic/home/xiangchunyang/software/OpenBLAS-0.3.23-install-ilp64/lib
   BOOST =/home/share/l6eub2ic/home/xiangchunyang/software/boost_1.80.0_install_openmpi_64
   LFLAGS = -L${BOOST}/lib -lboost_chrono-mt-a64 -lboost_timer-mt-a64 -lboost_serialization-mt-a64 -lboost_system-mt-a64 -lboost_iostreams-mt-a64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-a64
   endif
else ifeq ($(strip $(machine)), scv7260)
   MATHLIB =/data/apps/oneAPI/2022.2/mkl/latest/lib/intel64
   BOOST =/data/home/scv7260/run/xiangchunyang/boost_1_80_0_install
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
else ifeq ($(strip $(machine)), DCU_419)
   MATHLIB = /public/software/compiler/intel/oneapi/mkl/latest/lib/intel64
   BOOST = /public/home/ictapp_j/xiangchunyang/boost-1.80.0-install
   FLAGS += -D__HIP_PLATFORM_HCC__ -DHAVE_HIP -w
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
else ifeq ($(strip $(machine)), mac)
   MATHLIB = /opt/anaconda3/envs/work/lib
   BOOST = /Users/zhendongli/Desktop/documents_ZL/Codes/boost/install
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
endif
FLAGS += -std=c++17 ${INCLUDE_DIR} -I${BOOST}/include 
 
ifeq ($(strip $(USE_GCC)),yes)
   # GCC compiler
   ifeq ($(strip $(DEBUG)),yes)
      FLAGS += -DDEBUG -O0 -w #-Wall
   else
      FLAGS += -DNDEBUG -O2 -w #-Wall
   endif
   #FLAGS += -gdwarf-4 -gstrict-dwarf # dwarf error in ld
   ifeq ($(strip $(USE_MPI)),no)
      CXX = g++
      CC = gcc
      FLAGS += -DSERIAL
   else
      CXX = mpicxx
      CC = mpicc
   endif
else
   # Intel compiler
   ifeq ($(strip $(DEBUG)), yes)
      FLAGS += -DDEBUG -O0 -w #-Wall 
   else 
      FLAGS += -DNDEBUG -O2 -w #-Wall 
   endif 
   ifeq ($(strip $(USE_MPI)),no)
      CXX = icpc
      CC = icc
      FLAGS += -DSERIAL
   else
      CXX = mpiicpc
      CC = mpiicc
   endif
endif

ifeq ($(strip $(USE_BLAS)),no)
# OpenMP & MKL
   ifeq ($(strip $(USE_OPENMP)),no)
      ifeq ($(strip $(USE_ILP64)), no)
      # serial version of MKL
      MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
             -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
      else
      MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
             -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl -DMKL_ILP64 -m64
      endif
   else
   # parallel version of MKL
   # Use GNU OpenMP library: -lmkl_gnu_thread -lgomp replace -liomp5
      ifeq ($(strip $(USE_ILP64)), no)
      MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
             -lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl 
      else
	   # https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.sl42kc
      MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
             -lmkl_intel_ilp64 -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64
      endif

    	# special treatment for my mac machine
      ifeq ($(strip $(machine)), mac)
         FLAGS += -fopenmp 
         MATH += -lmkl_intel_thread -liomp5 
      else
         ifeq ($(strip $(USE_GCC)),yes)
            FLAGS += -fopenmp 
            MATH += -lmkl_gnu_thread -lgomp
         else
            FLAGS += -qopenmp 
            MATH += -lmkl_intel_thread -liomp5 
         endif
      endif
   endif
else
   # parallel version of MKL
   # Use GNU OpenMP library: -lmkl_gnu_thread -lgomp replace -liomp5
   MATH = \
	        -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
          -lopenblas_omp -lpthread -lm -ldl -lrt 
          #-lblas64 -llapack64 -lpthread -lm -ldl -lrt 
   FLAGS += -fopenmp -DUSE_BLAS -DLAPACK_ILP64 -DMKL_ILP64 -DOPENBLAS_USE64BITINT -DUSE64BITINT

endif
# quaternion matrix diagonalization
MATH += -L./extlibs/zquatev -lzquatev 
LFLAGS += ${MATH}

# GPU
ifeq ($(strip $(USE_GPU)), yes)
FLAGS += -I./src/ctns/gpu_kernel
LFLAGS += -L./src/ctns/gpu_kernel -lctnsGPU
ifeq ($(strip $(machine)), DCU_419)
   HIP_DIR=/public/software/compiler/rocm/rocm-3.3.0/hip
   MAGMA_DIR=/public/software/mathlib/magma/magma-rocm_3.3_develop
   FLAGS += -DGPU -DUSE_HIP -I${MAGMA_DIR}/include -I${HIP_DIR}/include 
   LFLAGS += -L${MAGMA_DIR}/lib -lmagma  -L${HIP_DIR}/lib -lhip_hcc -lhiprtc
else ifeq ($(strip $(machine)), scy0799)
   CUDA_DIR=/data/apps/cuda/11.2
   MAGMA_DIR=/data01/home/scy0799/run/xiangchunyang/project/magma-2.6.1-install/
   FLAGS += -DGPU -I${MAGMA_DIR}/include -I${CUDA_DIR}/include 
   LFLAGS += -L${MAGMA_DIR}/lib -lmagma -L${CUDA_DIR}/lib64 -lcudart_static
else ifeq ($(strip $(machine)), scv7260)
   CUDA_DIR=/data/apps/cuda/11.4
   MAGMA_DIR=/data/home/scv7260/run/xiangchunyang/magma_2_6_1_install
   FLAGS += -DGPU -I${MAGMA_DIR}/include -I${CUDA_DIR}/include
   LFLAGS += -L${MAGMA_DIR}/lib -lmagma -L${CUDA_DIR}/lib64 -lcudart_static
else ifeq ($(strip $(machine)), wuhan)
   CUDA_DIR=/home/HPCBase/compilers/cuda/11.4.0
	 MAGMA_DIR=/home/share/l6eub2ic/home/xiangchunyang/software/magma-2.7.1-install-openblas64
   FLAGS += -DGPU -I${MAGMA_DIR}/include -I${CUDA_DIR}/include
   LFLAGS += -L${MAGMA_DIR}/lib -lmagma -L${CUDA_DIR}/lib64 -lcudart_static -lcublas
   ifeq ($(strip $(USE_NCCL)), yes)
      NCCL_DIR = /home/HPCBase/libs/nccl/2.16.5-cuda11.4
      FLAGS += -DNCCL -I${NCCL_DIR}/include	
      LFLAGS += -L${NCCL_DIR}/lib -lnccl
   endif
else ifeq ($(strip $(machine)), dell2)
   CUDA_DIR= /home/dell/anaconda3/envs/pytorch
   MAGMA_DIR = ../magma/magma-2.6.1
   FLAGS += -DGPU -I${MAGMA_DIR}/include -I${CUDA_DIR}/include
   LFLAGS += -L${MAGMA_DIR}/lib -lmagma -L${CUDA_DIR}/lib -lcudart -lrt
   ifeq ($(strip $(USE_NCCL)), yes)
      NCCL_DIR = /home/dell/public-soft/nccl/build
      FLAGS += -DNCCL -I${NCCL_DIR}/include	
      LFLAGS += -L${NCCL_DIR}/lib -lnccl
   endif
else ifeq ($(strip $(machine)), dell)
   CUDA_DIR= /usr/local/cuda
   MAGMA_DIR = ../magma/magma-2.6.1
   FLAGS += -DGPU -I${MAGMA_DIR}/include -I${CUDA_DIR}/include
   LFLAGS += -L${MAGMA_DIR}/lib -lmagma -L${CUDA_DIR}/lib64 -lcudart_static
else ifeq ($(strip $(machine)), jiageng)
   CUDA_DIR= ${CUDADIR}
   MAGMA_DIR = /public/home/bnulizdtest/magma
   FLAGS += -DGPU -I${MAGMA_DIR}/include -I${CUDA_DIR}/include
   LFLAGS += -L${MAGMA_DIR}/lib -lmagma -L${CUDA_DIR}/lib64 -lcudart_static -lrt -lcublas
   ifeq ($(strip $(USE_NCCL)), yes)
      NCCL_DIR = /public/home/bnulizdtest/nccl/build
      FLAGS += -DNCCL -I${NCCL_DIR}/include
      LFLAGS += -L${NCCL_DIR}/lib -lnccl
   endif
endif
endif

# IO
ifeq ($(strip $(USE_LZ4)), yes)
   FLAGS += -DLZ4 -I./extlibs/lz4-dev/lib
   LFLAGS += -L./extlibs/lz4-dev/lib -llz4
endif
ifeq ($(strip $(USE_ZSTD)), yes)
   FLAGS += -DZSTD -I./extlibs/zstd-dev/lib
   LFLAGS += -L./extlibs/zstd-dev/lib -lzstd
endif

SRC = src
BIN_DIR = ./bin
OBJ_DIR = ./obj
LIB_DIR = ./lib

# all dependence
SRC_DIR_CORE = ./$(SRC)/core
SRC_DIR_IO   = ./$(SRC)/io

ifeq ($(strip $(INSTALL_CI)), yes)
   SRC_DIR_CI   = ./$(SRC)/ci
endif

ifeq ($(strip $(INSTALL_CTNS)), yes)
   SRC_DIR_QT   = ./$(SRC)/ctns/qtensor
   SRC_DIR_CTNS = ./$(SRC)/ctns
endif

ifeq ($(strip $(INSTALL_VMC)), yes)
   SRC_DIR_VMC  = ./$(SRC)/vmc
endif

SRC_DIR_EXPT = ./$(SRC)/experiment

INCLUDE_DIR = -I$(SRC_DIR_CORE) \
     	        -I$(SRC_DIR_IO) \
     	        -I$(SRC_DIR_CI) \
     	        -I$(SRC_DIR_QT) \
     	        -I$(SRC_DIR_CTNS) \
     	        -I$(SRC_DIR_VMC) \
     	        -I$(SRC_DIR_EXPT) 

ifeq ($(strip $(USE_GPU)), yes)
   SRC_DIR_GPU = ./$(SRC)/gpu
   INCLUDE_DIR += -I$(SRC_DIR_GPU)
endif

SRC_DEP = $(wildcard $(SRC_DIR_CORE)/*.cpp \
	  	     $(SRC_DIR_IO)/*.cpp  \
	  	     $(SRC_DIR_CI)/*.cpp \
	  	     $(SRC_DIR_QT)/*.cpp \
	  	     $(SRC_DIR_CTNS)/*.cpp \
	  	     $(SRC_DIR_VMC)/*.cpp \
	  	     $(SRC_DIR_EXPT)/*.cpp \
	  	     $(SRC_DIR_GPU)/*.cpp)

OBJ_DEP = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_DEP}))

# separate libraries
SRC_CORE = $(wildcard $(SRC_DIR_CORE)/*.cpp)
OBJ_CORE = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_CORE}))

SRC_IO = $(wildcard $(SRC_DIR_IO)/*.cpp) 
OBJ_IO = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_IO}))

SRC_CI = $(wildcard $(SRC_DIR_CI)/*.cpp)
OBJ_CI = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_CI}))

SRC_CTNS = $(wildcard $(SRC_DIR_QT)/*.cpp \
		      $(SRC_DIR_CTNS)/*.cpp \
		      $(SRC_DIR_EXPT)/*.cpp \
		      $(SRC_DIR_GPU)/*.cpp)
OBJ_CTNS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_CTNS}))

SRC_VMC = $(wildcard $(SRC_DIR_VMC)/*.cpp)
OBJ_VMC = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_VMC}))

# all the files with main functions
SRC_ALL = $(SRC_DEP) 
SRC_ALL += $(wildcard ./$(SRC)/drivers/*.cpp)
OBJ_ALL = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_ALL}))

all: depend core ci ctns vmc

core: $(LIB_DIR)/libcore.a $(LIB_DIR)/libio.a $(BIN_DIR)/tests_core.x 

ifeq ($(strip $(INSTALL_CI)), yes)
ci: $(LIB_DIR)/libci.a $(BIN_DIR)/tests_ci.x $(BIN_DIR)/exactdiag.x $(BIN_DIR)/fci.x $(BIN_DIR)/sci.x
else
ci:
endif

ifeq ($(strip $(INSTALL_CTNS)), yes)
ctns: $(LIB_DIR)/libctns.a $(BIN_DIR)/tests_ctns.x $(BIN_DIR)/prectns.x $(BIN_DIR)/ctns.x
else
ctns:	
endif

ifeq ($(strip $(INSTALL_VMC)), yes)
vmc: $(LIB_DIR)/libvmc.a $(BIN_DIR)/vmc.x
else
vmc:
endif

# version
GIT_HASH=`git rev-parse HEAD`
FLAGS += -DGIT_HASH="\"$(GIT_HASH)\"" 

depend:
	echo "Check compilation options:"; \
	echo " machine = " $(machine); \
	echo " DEBUG = " $(DEBUG); \
	echo " USE_GCC = " $(USE_GCC); \
	echo " USE_MPI = " $(USE_MPI); \
	echo " USE_OPENMP = " $(USE_OPENMP); \
	echo " USE_BLAS = " $(USE_BLAS); \
	echo " USE_ILP64 = " $(USE_ILP64); \
	echo " USE_GPU = " $(USE_GPU); \
	echo " CXX = " $(CXX); \
	echo " CC = " $(CC); \
	set -e; \
	mkdir -p $(BIN_DIR) $(OBJ_DIR) $(LIB_DIR); \
	echo $(SRC_ALL); \
	$(CXX) $(FLAGS) -I${BOOST}/include -MM $(SRC_ALL) > $$$$.depend; \
	sed 's,\([^.]*\.o\),$(OBJ_DIR)/\1,' < $$$$.depend > .depend; \
	echo 'finish dependency check!'; \
	rm -f $$$$.depend # $$$$ id number 
-include .depend

# LIBARARIES
$(LIB_DIR)/libcore.a: $(OBJ_CORE)
	@echo "=== COMPLIE $@"
	ar crv $@ $^

$(LIB_DIR)/libio.a: $(OBJ_IO)
	@echo "=== COMPLIE $@"
	ar crv $@ $^

$(LIB_DIR)/libci.a: $(OBJ_CI) $(OBJ_CORE) $(OBJ_IO)
	@echo "=== COMPLIE $@"
	ar crv $@ $^

$(LIB_DIR)/libctns.a: $(OBJ_CTNS) $(OBJ_CI) $(OBJ_CORE) $(OBJ_IO)
	@echo "=== COMPLIE $@"
	ar crv $@ $^

$(LIB_DIR)/libvmc.a: $(OBJ_VMC) $(OBJ_CI) $(OBJ_CORE) $(OBJ_IO)
	@echo "=== COMPLIE $@"
	ar crv $@ $^

# Executables
$(BIN_DIR)/tests_core.x: $(OBJ_DIR)/tests_core.o $(LIB_DIR)/libcore.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/tests_core.o $(LFLAGS) -L$(LIB_DIR) -lcore

$(BIN_DIR)/tests_ci.x: $(OBJ_DIR)/tests_ci.o $(LIB_DIR)/libci.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/tests_ci.o $(LFLAGS) -L$(LIB_DIR) -lci

$(BIN_DIR)/tests_ctns.x: $(OBJ_DIR)/tests_ctns.o $(LIB_DIR)/libctns.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/tests_ctns.o -L$(LIB_DIR) -lctns $(LFLAGS)

# Main:
$(BIN_DIR)/exactdiag.x: $(OBJ_DIR)/exactdiag.o $(LIB_DIR)/libcore.a $(LIB_DIR)/libio.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/exactdiag.o $(LFLAGS) -L$(LIB_DIR) -lcore -lio

$(BIN_DIR)/fci.x: $(OBJ_DIR)/fci.o $(LIB_DIR)/libci.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/fci.o $(LFLAGS) -L$(LIB_DIR) -lci

$(BIN_DIR)/sci.x: $(OBJ_DIR)/sci.o $(LIB_DIR)/libci.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/sci.o $(LFLAGS) -L$(LIB_DIR) -lci

$(BIN_DIR)/prectns.x: $(OBJ_DIR)/prectns.o $(LIB_DIR)/libctns.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/prectns.o -L$(LIB_DIR) -lctns $(LFLAGS) 

$(BIN_DIR)/ctns.x: $(OBJ_DIR)/ctns.o $(LIB_DIR)/libctns.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/ctns.o -L$(LIB_DIR) -lctns $(LFLAGS) 

$(BIN_DIR)/vmc.x: $(OBJ_DIR)/vmc.o $(LIB_DIR)/libvmc.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/vmc.o $(LFLAGS) -L$(LIB_DIR) -lci

# Needs to be here! 
$(OBJ_ALL):
	@echo "=== COMPILE $@ FROM $<" # just from *.cpp is sufficient	
	$(CXX) $(FLAGS) -o $@ -c $< $(LFLAGS)

clean:
	rm -f obj/*.o
	rm -f bin/*.x
	rm -f lib/*.a
	rm -f *.depend
