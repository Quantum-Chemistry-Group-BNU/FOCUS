
machine = mac #dell #lenovo

DEBUG = yes
USE_GCC = yes
USE_MPI = yes
USE_OPENMP = yes
# compression
USE_LZ4 = no
USE_ZSTD = no
USE_GPU = no #yes

# set library
ifeq ($(strip $(machine)), lenovo)
   MATHLIB = /opt/intel/oneapi/mkl/2022.0.2/lib/intel64
   BOOST = /home/lx.aftware/boost/install_1_79_0
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
else ifeq ($(strip $(machine)), mac)
   #MATHLIB = /Users/zhendongli/anaconda2/envs/py38/lib
   #BOOST = /Users/zhendongli/Desktop/FOCUS_program/boost/install
   MATHLIB = /opt/anaconda3/envs/work/lib
   BOOST = /Users/zhendongli/Desktop/documents_ZL/Codes/boost/install
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_chrono-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64 -lboost_iostreams-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
endif
FLAGS = -std=c++17 ${INCLUDE_DIR} -I${BOOST}/include #-gdwarf-4 -gstrict-dwarf # dwarf error in ld
 
ifeq ($(strip $(USE_GCC)),yes)
   # GCC compiler
   ifeq ($(strip $(DEBUG)),yes)
      FLAGS += -DDEBUG -g -O0 -Wall 
   else
      FLAGS += -DNDEBUG -O2 -Wall 
   endif
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
      FLAGS += -DDEBUG -g -O0 -Wall 
   else 
      FLAGS += -DNDEBUG -O2 -Wall 
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

ifeq ($(strip $(USE_OPENMP)),no)
   # serial version of MKL
   MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
          -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
   # mac framework Accelerate
   #MATH = -llapack -lblas 
else
   ifeq ($(strip $(USE_GCC)),yes)
      FLAGS += -fopenmp
   else
      FLAGS += -qopenmp	
   endif
   # https:/.aftware.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-adv.ar.html	
   # parallel version of MKL
   MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
          -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl \
   	  -liomp5
   # Use GNU OpenMP library: -lmkl_gnu_thread -lgomp replace -liomp5
endif
# quaternion matrix diagonalization
MATH += -L./extlibs/zquatev -lzquatev
LFLAGS += ${MATH}

# IO
ifeq ($(strip $(USE_LZ4)), yes)
   FLAGS += -DLZ4 -I./extlibs/lz4-dev/lib
   LFLAGS += -L./extlibs/lz4-dev/lib -llz4
endif
ifeq ($(strip $(USE_ZSTD)), yes)
   FLAGS += -DZSTD -I./extlibs/zstd-dev/lib
   LFLAGS += -L./extlibs/zstd-dev/lib -lzstd
endif

# GPU
ifeq ($(strip $(USE_GPU)), yes)
   CUDA_DIR= /usr/local/cuda
   MAGMA_DIR = ../magma/install
   FLAGS += -DGPU -I${MAGMA_DIR}/include -I${CUDA_DIR}/include
   LFLAGS += -L${MAGMA_DIR}/lib -lmagma -lmagma_sparse -L${CUDA_DIR}/lib64 -lcudart_static
endif

SRC = src
BIN_DIR = ./bin
OBJ_DIR = ./obj
LIB_DIR = ./lib

# all dependence
SRC_DIR_CORE = ./$(SRC)/core
SRC_DIR_IO   = ./$(SRC)/io
SRC_DIR_CI   = ./$(SRC)/ci
SRC_DIR_QT   = ./$(SRC)/ctns/qte.ar
SRC_DIR_CTNS = ./$(SRC)/ctns
SRC_DIR_VMC  = ./$(SRC)/vmc
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

all: depend \
     $(LIB_DIR)/libcore.a \
     $(LIB_DIR)/libio.a \
     $(LIB_DIR)/libci.a \
     $(LIB_DIR)/libctns.a \
     $(LIB_DIR)/libvmc.a \
     $(BIN_DIR)/tests_core.x \
     $(BIN_DIR)/tests_ci.x \
     $(BIN_DIR)/tests_ctns.x \
     $(BIN_DIR)/exactdiag.x \
     $(BIN_DIR)/fci.x \
     $(BIN_DIR)/sci.x \
     $(BIN_DIR)/prectns.x \
     $(BIN_DIR)/ctns.x \
     $(BIN_DIR)/vmc.x \

# version
GIT_HASH=`git rev-parse HEAD`
FLAGS += -DGIT_HASH="\"$(GIT_HASH)\"" 

depend:
	echo "Check compilation options:"; \
	echo " machine = " $(machine); \
	echo " DEBUG = " $(DEBUG); \
	echo " USE_GCC = " $(USE_GCC); \
	echo " USE_OPENMP = " $(USE_OPENMP); \
	echo " USE_MPI = " $(USE_MPI); \
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
	$(CXX) $(FLAGS) -o $@ -c $< 

clean:
	rm -f obj/*.o
	rm -f bin/*.x
	rm -f lib/*.a
	rm -f *.depend
