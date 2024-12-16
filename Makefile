
machine = dell2 #jiageng #scv7260 #scy0799 #DCU_419 #mac #dell #lenovo

DEBUG = yes
USE_GCC = yes
USE_MPI = yes
USE_OPENMP = yes
USE_MKL = yes
USE_ILP64 = yes
USE_GPU = yes
USE_NCCL = yes
USE_TCMALLOC = yes
USE_MAGMA = no #yes
# compression
USE_LZ4 = no
USE_ZSTD = no
# exec
INSTALL_CI = yes
INSTALL_CTNS = yes
INSTALL_POST = no #yes
INSTALL_VMC = no #yes
INSTALL_PY = no #yes

# set library
ifeq ($(strip $(machine)), lenovo)
   MATHLIB = /opt/intel/oneapi/mkl/2022.0.2/lib/intel64
   BOOST = /home/lx/software/boost/install_1_79_0
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
   NLOPTDIR_LIB = ./extlibs/nlopt-2.7.1/build/install/lib64
   NLOPTDIR_INCLUDE = ./extlibs/nlopt-2.7.1/build/install/include 

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
   GSLDIR = /usr/local
   NLOPTDIR_LIB = ./extlibs/nlopt-2.7.1/build/install-lzd/lib64
   NLOPTDIR_INCLUDE = ./extlibs/nlopt-2.7.1/build/install-lzd/include
   ifeq ($(strip $(USE_TCMALLOC)), yes)  
      FLAGS += -DTCMALLOC -I/usr/local/include/gperftools
      LFLAGS += -L/usr/local/lib -ltcmalloc
   endif

else ifeq ($(strip $(machine)), jinan)
   MATHLIB = ${MKLROOT}/lib
   BOOST = ../boost/install
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
   GSLDIR = /usr/local
   NLOPTDIR_LIB = /usr/local/lib
   NLOPTDIR_INCLUDE = /usr/local/include
   ifeq ($(strip $(USE_TCMALLOC)), yes)  
      FLAGS += -DTCMALLOC -I/usr/include/google
      LFLAGS += -L/usr/lib/x86_64-linux-gnu -ltcmalloc
   endif

else ifeq ($(strip $(machine)), jiageng)
   MATHLIB = ./mkl2022 #/public/software/intel/oneapi2021/mkl/latest #/public/software/anaconda/anaconda3-2022.5/lib
   BOOST = /public/home/bnulizdtest/boost/install
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
   #FLAGS += -no-multibyte-chars
   GSLDIR = ./extlibs/gsl-install
   NLOPTDIR_LIB = ./extlibs/nlopt-2.7.1/build/lzdnlopt/lib64
   NLOPTDIR_INCLUDE = ./extlibs/nlopt-2.7.1/build/lzdnlopt/include
   ifeq ($(strip $(USE_TCMALLOC)), yes)
      FLAGS += -DTCMALLOC -I./extlibs/gperftools-master/lzdtcmalloc/include/gperftools
      LFLAGS += -L./extlibs/gperftools-master/lzdtcmalloc/lib -ltcmalloc
   endif

else ifeq ($(strip $(machine)), a800_xiyun)
   MATHLIB = ${MKLROOT}
   BOOST = /storage/boost/install
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
   #FLAGS += -no-multibyte-chars
   GSLDIR = ./extlibs/gsl-install
   NLOPTDIR_LIB = /storage/FOCUS/extlibs/nlopt-2.7.1/build/lib
   NLOPTDIR_INCLUDE = /storage/FOCUS/extlibs/nlopt-2.7.1/build/include

else ifeq ($(strip $(machine)), a800)
   MATHLIB = $(MKLROOT)
   BOOST = /GLOBALFS/bnu_pp_1/boost/install
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
   GSLDIR = /GLOBALFS/bnu_pp_1/FOCUS/extlibs/gsl-2.7.1/lzdgsl

else ifeq ($(strip $(machine)), scy0799)
   MATHLIB =/data/apps/OneApi/2022.1/oneapi/mkl/latest/lib/intel64/
   BOOST =/data01/home/scy0799/run/xiangchunyang/project/boost_1_80_0_install
   LFLAGS = -L${BOOST}/lib -lboost_chrono-mt-x64 -lboost_timer-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
else ifeq ($(strip $(machine)), wuhan)
   #MATHLIB =/home/HPCBase/libs/openblas0.3.18_kgcc9.3.1/libs
   #MATHLIB = /home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/lapack-3.11.0-install-64/lib64
   MATHLIB = /home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/OpenBLAS-0.3.23-install-ilp64/lib
   BOOST =/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/boost_1.80.0_install_openmpi_64
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
else ifeq ($(strip $(machine)), mac)
   #MATHLIB = /Users/zhendongli/Desktop/FOCUS/mathlib/lapack-3.12.0/build/lib
   MATHLIB = /Users/zhendongli/anaconda3/envs/osx64test/lib
   BOOST = /Users/zhendongli/Desktop/documents_ZL/Codes/boost/install_1_83_0
   LFLAGS = -L${BOOST}/lib -Wl,-rpath,${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
   GSLDIR = /usr/local
   NLOPTDIR_LIB = /usr/local/lib
   NLOPTDIR_INCLUDE = /usr/local/include
   ifeq ($(strip $(USE_TCMALLOC)), yes) 
      FLAGS += -DTCMALLOC -I/usr/local/include/gperftools
      LFLAGS += -L/usr/local/lib -ltcmalloc
   endif

else ifeq ($(strip $(machine)), archlinux)
   MATHLIB = /opt/intel/oneapi/mkl/2023.1.0/lib/intel64
   BOOST = /usr
   LFLAGS = -lboost_timer -lboost_chrono -lboost_serialization -lboost_system -lboost_iostreams
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi
   endif
endif
LFLAGS += -L${GSLDIR}/lib -lgsl -L${NLOPTDIR_LIB} -lnlopt
FLAGS += -std=c++17 ${INCLUDE_DIR} -I${BOOST}/include -I${GSLDIR}/include -I${NLOPTDIR_INCLUDE}
 
target = depend core ci ctns vmc
ifeq ($(strip $(INSTALL_PY)), yes)
   ifeq ($(strip $(machine)), mac)
      PYBIND = -undefined dynamic_lookup 
   endif
   PYBIND += $(shell python3 -m pybind11 --includes)
   FLAGS += $(PYBIND) -DPYTHON_BINDING
   FLAGS += -fPIC
   target += python 
endif

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

# special treatment for my mac machine
ifeq ($(strip $(machine)), mac)
   #MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
   #       -lblas64 -llapack64 -lpthread -lgfortran
   MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) -lmkl_intel_ilp64 -lmkl_intel_thread -liomp5 -lmkl_core -lpthread -lm -ldl
   FLAGS += -DUSE_MKL -DMKL_ILP64 -m64 -fopenmp 
else
ifeq ($(strip $(USE_MKL)),yes)
	# FLAGS
   FLAGS += -DUSE_MKL
   ifeq ($(strip $(USE_ILP64)),yes)
      FLAGS += -DMKL_ILP64 -m64
   endif
	# LFLAGS
   MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB)
   ifeq ($(strip $(USE_ILP64)),yes)
      MATH += -lmkl_intel_ilp64 
   else
      MATH += -lmkl_intel_lp64 
   endif
	# OpenMP & MKL
   ifeq ($(strip $(USE_OPENMP)),no)
      MATH += -lmkl_sequential
   else
      ifeq ($(strip $(USE_GCC)),yes)
         FLAGS += -fopenmp 
         MATH += -lmkl_gnu_thread -lgomp
      else
         FLAGS += -qopenmp 
         MATH += -lmkl_intel_thread -liomp5 
      endif
   endif
   MATH += -lmkl_core -lpthread -lm -ldl
else
   # openblas/kblas
   MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
          -lopenblas -lpthread -lm -ldl -lrt 
          #-lblas64 -llapack64 -lpthread -lm -ldl -lrt 
   FLAGS += -fopenmp -DLAPACK_ILP64 -DMKL_ILP64 -DOPENBLAS_USE64BITINT -DUSE64BITINT
endif
endif
# quaternion matrix diagonalization
MATH += -L./extlibs/zquatev -lzquatev 
LFLAGS += ${MATH}

# GPU
ifeq ($(strip $(USE_GPU)), yes)

ifeq ($(strip $(machine)), wuhan)
   CUDA_DIR=/home/HPCBase/compilers/cuda/11.4.0
   FLAGS += -DGPU -I${CUDA_DIR}/include
   LFLAGS += -L${CUDA_DIR}/lib64 -lcudart_static -lcublas -lcusolver
   ifeq ($(strip $(USE_MAGMA)), yes)
      MAGMA_DIR=/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/magma-2.7.1-install-openblas64
      FLAGS += -DMAGMA -I${MAGMA_DIR}/include 
      LFLAGS += -L${MAGMA_DIR}/lib -lmagma
   endif
   ifeq ($(strip $(USE_NCCL)), yes)
      NCCL_DIR = /home/HPCBase/libs/nccl/2.16.5-cuda11.4
      FLAGS += -DNCCL -I${NCCL_DIR}/include	
      LFLAGS += -L${NCCL_DIR}/lib -lnccl
   endif

else ifeq ($(strip $(machine)), dell2)
   CUDA_DIR= /home/dell/anaconda3/envs/pytorch2
   FLAGS += -DGPU -I${CUDA_DIR}/include
   LFLAGS += -L${CUDA_DIR}/lib -lcudart -lrt -lcublas -lcusolver
   ifeq ($(strip $(USE_MAGMA)), yes)
      MAGMA_DIR = ../magma/magma-2.6.1
      FLAGS += -DMAGMA -I${MAGMA_DIR}/include 
      LFLAGS += -L${MAGMA_DIR}/lib -lmagma
   endif
   ifeq ($(strip $(USE_NCCL)), yes)
      NCCL_DIR = /home/dell/public-soft/nccl/build
      FLAGS += -DNCCL -I${NCCL_DIR}/include	
      LFLAGS += -L${NCCL_DIR}/lib -lnccl
   endif

else ifeq ($(strip $(machine)), dell)
   CUDA_DIR= /usr/local/cuda
   FLAGS += -DGPU -I${CUDA_DIR}/include
   LFLAGS += -L${CUDA_DIR}/lib64 -lcudart_static -lcusolver
   ifeq ($(strip $(USE_MAGMA)), yes)
      MAGMA_DIR = ../magma/magma-2.6.1
      FLAGS += -DMAGMA -I${MAGMA_DIR}/include 
      LFLAGS += -L${MAGMA_DIR}/lib -lmagma
   endif
 
else ifeq ($(strip $(machine)), jinan)
   CUDA_DIR= ${CUDADIR}
   FLAGS += -DGPU -I${CUDA_DIR}/include
   LFLAGS += -L${CUDA_DIR}/lib64 -lcudart_static -lrt -lcublas -ldl -lcusolver
   ifeq ($(strip $(USE_MAGMA)), yes)
      MAGMA_DIR = ../magma-2.6.1
      FLAGS += -DMAGMA -I${MAGMA_DIR}/include 
      LFLAGS += -L${MAGMA_DIR}/lib -lmagma
   endif 
   ifeq ($(strip $(USE_NCCL)), yes)
      NCCL_DIR = /yeesuanAI03/lzd/nccl-master/build
      FLAGS += -DNCCL -I${NCCL_DIR}/include
      LFLAGS += -L${NCCL_DIR}/lib -lnccl
   endif

else ifeq ($(strip $(machine)), jiageng)
   CUDA_DIR= ${CUDADIR}
   FLAGS += -DGPU -I${CUDA_DIR}/include
   LFLAGS += -L${CUDA_DIR}/lib64 -lcudart_static -lrt -lcublas -lcusolver
   ifeq ($(strip $(USE_MAGMA)), yes)
      MAGMA_DIR = /public/home/bnulizdtest/magma
      FLAGS += -DMAGMA -I${MAGMA_DIR}/include 
      LFLAGS += -L${MAGMA_DIR}/lib -lmagma
   endif 
   ifeq ($(strip $(USE_NCCL)), yes)
      NCCL_DIR = /public/home/bnulizdtest/nccl/build
      FLAGS += -DNCCL -I${NCCL_DIR}/include
      LFLAGS += -L${NCCL_DIR}/lib -lnccl
   endif

else ifeq ($(strip $(machine)), a800_xiyun)
   CUDA_DIR= ${CUDA_ROOT}
   FLAGS += -DGPU -I${CUDA_DIR}/include -DGEMMGROUPED
   LFLAGS += -L${CUDA_DIR}/lib64 -lcudart_static -lrt -lcublas -lcusolver
   ifeq ($(strip $(USE_MAGMA)), yes)
      MAGMA_DIR = /storage/FOCUS/extlibs/magma-2.8.0
      FLAGS += -DMAGMA -I${MAGMA_DIR}/include 
      LFLAGS += -L${MAGMA_DIR}/lib -lmagma
   endif 
   ifeq ($(strip $(USE_NCCL)), yes)
      NCCL_DIR = /root/nccl_apps/nccl220
      FLAGS += -DNCCL -I${NCCL_DIR}/include
      LFLAGS += -L${NCCL_DIR}/lib -lnccl
   endif

endif

FLAGS += -I./src/ctns/gpu_kernel
LFLAGS += -L./src/ctns/gpu_kernel -lctnsGPU
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
SRC_DIR_EXPT = ./$(SRC)/experiment

INCLUDE_DIR = -I./src \
				  -I$(SRC_DIR_CORE) \
     	        -I$(SRC_DIR_IO) \
              -I$(SRC_DIR_EXPT)

ifeq ($(strip $(INSTALL_CI)), yes)
   SRC_DIR_CI   = ./$(SRC)/ci
   INCLUDE_DIR += -I$(SRC_DIR_CI)
endif

ifeq ($(strip $(INSTALL_CTNS)), yes)
   SRC_DIR_QT   = ./$(SRC)/qtensor
   SRC_DIR_CTNS = ./$(SRC)/ctns
   INCLUDE_DIR += -I$(SRC_DIR_QT)
   INCLUDE_DIR += -I$(SRC_DIR_CTNS)
endif

ifeq ($(strip $(INSTALL_POST)), yes)
   SRC_DIR_POST = ./$(SRC)/post
   INCLUDE_DIR += -I$(SRC_DIR_POST)
endif

ifeq ($(strip $(INSTALL_VMC)), yes)
   SRC_DIR_VMC  = ./$(SRC)/vmc
   INCLUDE_DIR += -I$(SRC_DIR_VMC)
endif

ifeq ($(strip $(INSTALL_PY)), yes)
   SRC_DIR_PY = ./$(SRC)/python
   INCLUDE_DIR += -I$(SRC_DIR_PY)
endif

ifeq ($(strip $(USE_GPU)), yes)
   SRC_DIR_GPU = ./$(SRC)/gpu
   INCLUDE_DIR += -I$(SRC_DIR_GPU)
endif

SRC_DEP = $(wildcard $(SRC_DIR_CORE)/*.cpp \
	  	     $(SRC_DIR_IO)/*.cpp  \
	  	     $(SRC_DIR_CI)/*.cpp \
	  	     $(SRC_DIR_QT)/*.cpp \
	  	     $(SRC_DIR_CTNS)/*.cpp \
	  	     $(SRC_DIR_POST)/*.cpp \
	  	     $(SRC_DIR_VMC)/*.cpp \
	  	     $(SRC_DIR_EXPT)/*.cpp \
	  	     $(SRC_DIR_GPU)/*.cpp \
           $(SRC_DIR_PY)/*.cpp)

OBJ_DEP = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_DEP}))

# separate libraries
SRC_CORE = $(wildcard $(SRC_DIR_CORE)/*.cpp \
				 $(SRC_DIR_GPU)/*.cpp) # put GPU into CORE
OBJ_CORE = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_CORE}))

SRC_IO = $(wildcard $(SRC_DIR_IO)/*.cpp) 
OBJ_IO = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_IO}))

SRC_CI = $(wildcard $(SRC_DIR_CI)/*.cpp)
OBJ_CI = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_CI}))

SRC_CTNS = $(wildcard $(SRC_DIR_QT)/*.cpp \
		      $(SRC_DIR_CTNS)/*.cpp \
		      $(SRC_DIR_EXPT)/*.cpp)
OBJ_CTNS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_CTNS}))

SRC_POST = $(wildcard $(SRC_DIR_POST)/*.cpp)
OBJ_POST = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_POST}))

SRC_VMC = $(wildcard $(SRC_DIR_VMC)/*.cpp)
OBJ_VMC = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_VMC}))

SRC_PY = $(wildcard $(SRC_DIR_PY)/*.cpp)
OBJ_PY = $(patsubst %.cpp,$(OBJ_DIR)/%.o, $(notdir ${SRC_PY}))

# all the files with main functions
SRC_ALL = $(SRC_DEP) 
SRC_ALL += $(wildcard $(SRC)/drivers/*.cpp \
			  $(SRC)/drivers/benchmark/*.cpp \
			  $(SRC)/drivers/tests/*.cpp)
OBJ_ALL = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_ALL}))

all: $(target) 

core: $(LIB_DIR)/libcore.a \
	$(LIB_DIR)/libio.a \
	$(BIN_DIR)/tests_core.x \
	$(BIN_DIR)/benchmark_mathlib.x $(BIN_DIR)/benchmark_blas.x \
	$(BIN_DIR)/benchmark_io.x $(BIN_DIR)/benchmark_mpi.x \
	$(BIN_DIR)/benchmark_nccl.x $(BIN_DIR)/benchmark_lapack.x 

ifeq ($(strip $(INSTALL_CI)), yes)
ci: $(LIB_DIR)/libci.a $(BIN_DIR)/tests_ci.x $(BIN_DIR)/exactdiag.x $(BIN_DIR)/fci.x $(BIN_DIR)/sci.x
else
ci:
endif

ifeq ($(strip $(INSTALL_CTNS)), yes)
ctns: $(LIB_DIR)/libctns.a $(BIN_DIR)/tests_ctns.x \
	$(BIN_DIR)/tests_oper.x $(BIN_DIR)/preprocess.x \
	$(BIN_DIR)/ctns.x $(BIN_DIR)/sadmrg.x $(BIN_DIR)/rdm.x
else
ctns:	
endif

ifeq ($(strip $(INSTALL_POST)), yes)
ctns: $(LIB_DIR)/libpost.a $(BIN_DIR)/post.x
else
ctns:	
endif

ifeq ($(strip $(INSTALL_VMC)), yes)
vmc: $(LIB_DIR)/libvmc.a $(BIN_DIR)/vmc.x
else
vmc:
endif


ifeq ($(strip $(INSTALL_PY)), yes)
python: $(LIB_DIR)/libbindpy.a $(LIB_DIR)/qubic.so
else
python:	
endif

# version
GIT_HASH=`git rev-parse HEAD`
FLAGS += -DGIT_HASH="\"$(GIT_HASH)\""  # print git log

depend:
	echo "Check compilation options:"; \
	echo " machine = " $(machine); \
	echo " DEBUG = " $(DEBUG); \
	echo " USE_GCC = " $(USE_GCC); \
	echo " USE_MPI = " $(USE_MPI); \
	echo " USE_OPENMP = " $(USE_OPENMP); \
	echo " USE_MKL = " $(USE_MKL); \
	echo " USE_ILP64 = " $(USE_ILP64); \
	echo " USE_GPU = " $(USE_GPU); \
	echo " USE_NCCL = " $(USE_NCCL); \
	echo " USE_MAGMA = " $(USE_MAGMA); \
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

$(LIB_DIR)/libpost.a: $(OBJ_POST) $(OBJ_CI) $(OBJ_CORE) $(OBJ_IO) $(OBJ_CTNS)
	@echo "=== COMPLIE $@"
	ar crv $@ $^

$(LIB_DIR)/libvmc.a: $(OBJ_VMC) $(OBJ_CI) $(OBJ_CORE) $(OBJ_IO)
	@echo "=== COMPLIE $@"
	ar crv $@ $^

$(LIB_DIR)/libbindpy.a: $(OBJ_PY) $(OBJ_CORE) $(OBJ_CI) $(OBJ_POST) $(OBJ_CTNS) $(OBJ_IO)
	@echo "=== COMPLIE $@"
	ar crv $@ $^

# Executables
$(BIN_DIR)/tests_core.x: $(OBJ_DIR)/tests_core.o $(LIB_DIR)/libcore.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/tests_core.o -L$(LIB_DIR) -lcore $(LFLAGS) 

$(BIN_DIR)/benchmark_mathlib.x: $(OBJ_DIR)/benchmark_mathlib.o $(LIB_DIR)/libcore.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/benchmark_mathlib.o -L$(LIB_DIR) -lcore $(LFLAGS) 

$(BIN_DIR)/benchmark_blas.x: $(OBJ_DIR)/benchmark_blas.o $(LIB_DIR)/libcore.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/benchmark_blas.o -L$(LIB_DIR) -lcore $(LFLAGS) 

$(BIN_DIR)/benchmark_io.x: $(OBJ_DIR)/benchmark_io.o $(LIB_DIR)/libcore.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/benchmark_io.o -L$(LIB_DIR) -lcore $(LFLAGS) 

$(BIN_DIR)/benchmark_mpi.x: $(OBJ_DIR)/benchmark_mpi.o $(LIB_DIR)/libcore.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/benchmark_mpi.o -L$(LIB_DIR) -lcore $(LFLAGS) 

$(BIN_DIR)/benchmark_nccl.x: $(OBJ_DIR)/benchmark_nccl.o $(LIB_DIR)/libcore.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/benchmark_nccl.o -L$(LIB_DIR) -lcore $(LFLAGS) 

$(BIN_DIR)/benchmark_lapack.x: $(OBJ_DIR)/benchmark_lapack.o $(LIB_DIR)/libcore.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/benchmark_lapack.o -L$(LIB_DIR) -lcore $(LFLAGS)

# CI
$(BIN_DIR)/tests_ci.x: $(OBJ_DIR)/tests_ci.o $(LIB_DIR)/libci.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/tests_ci.o -L$(LIB_DIR) -lci $(LFLAGS) 

# CTNS
$(BIN_DIR)/tests_ctns.x: $(OBJ_DIR)/tests_ctns.o $(LIB_DIR)/libctns.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/tests_ctns.o -L$(LIB_DIR) -lctns $(LFLAGS)

$(BIN_DIR)/tests_oper.x: $(OBJ_DIR)/tests_oper.o $(LIB_DIR)/libctns.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/tests_oper.o -L$(LIB_DIR) -lctns $(LFLAGS)

# Main:
$(BIN_DIR)/exactdiag.x: $(OBJ_DIR)/exactdiag.o $(LIB_DIR)/libcore.a $(LIB_DIR)/libio.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/exactdiag.o -L$(LIB_DIR) -lcore -lio $(LFLAGS) 

$(BIN_DIR)/fci.x: $(OBJ_DIR)/fci.o $(LIB_DIR)/libci.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/fci.o -L$(LIB_DIR) -lci $(LFLAGS) 

$(BIN_DIR)/sci.x: $(OBJ_DIR)/sci.o $(LIB_DIR)/libci.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/sci.o -L$(LIB_DIR) -lci $(LFLAGS) 

$(BIN_DIR)/preprocess.x: $(OBJ_DIR)/preprocess.o $(LIB_DIR)/libctns.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/preprocess.o -L$(LIB_DIR) -lctns $(LFLAGS) 

$(BIN_DIR)/ctns.x: $(OBJ_DIR)/ctns.o $(LIB_DIR)/libctns.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/ctns.o -L$(LIB_DIR) -lctns $(LFLAGS) 

$(BIN_DIR)/sadmrg.x: $(OBJ_DIR)/sadmrg.o $(LIB_DIR)/libctns.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/sadmrg.o -L$(LIB_DIR) -lctns $(LFLAGS) 

$(BIN_DIR)/rdm.x: $(OBJ_DIR)/rdm.o $(LIB_DIR)/libctns.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/rdm.o -L$(LIB_DIR) -lctns $(LFLAGS) 

$(BIN_DIR)/post.x: $(OBJ_DIR)/post.o $(LIB_DIR)/libpost.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/post.o -L$(LIB_DIR) -lpost $(LFLAGS) 

$(BIN_DIR)/vmc.x: $(OBJ_DIR)/vmc.o $(LIB_DIR)/libvmc.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/vmc.o -L$(LIB_DIR) -lci $(LFLAGS) 

# bind python
$(LIB_DIR)/qubic.so: $(OBJ_DIR)/bind_python.o $(LIB_DIR)/libbindpy.a
	@echo "=== LINK $@"
	$(CXX) $(FLAGS) -o $@ $(OBJ_DIR)/bind_python.o -shared $(LFLAGS) -L$(LIB_DIR) -lbindpy

# Needs to be here! 
$(OBJ_ALL):
	@echo "=== COMPILE $@ FROM $<" # just from *.cpp is sufficient	
	$(CXX) $(FLAGS) -o $@ -c $< $(LFLAGS)

clean:
	rm -f obj/*.o
	rm -f bin/*.x
	rm -f lib/*.a
	rm -f lib/*.so
	rm -f *.depend
