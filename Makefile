
machine = lenovo

DEBUG = no #yes
USE_GCC = no #yes
USE_MPI = yes
USE_OPENMP = yes

# set library
ifeq ($(strip $(machine)), lenovo)
   MATHLIB = /opt/intel/oneapi/mkl/2022.0.2/lib/intel64
   #HDF5 = /home/lx/lzd/hdf5/CMake-hdf5-1.12.1/HDF_Group/HDF5/1.12.1
   HDF5 = /home/lx/lzd/hdf5/hdf5-1.12.1/hdf5
   BOOST = /home/lx/software/boost/install_1_75_0
   LFLAGS = -L${BOOST}/lib -lboost_timer-mt-x64 -lboost_serialization-mt-x64 -lboost_system-mt-x64
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi-mt-x64
   endif
else
   MATHLIB = /Users/zhendongli/anaconda2/envs/py38/lib
   HDF5 = /usr/local
   BOOST = /usr/local
   LFLAGS = -L${BOOST}/lib -lboost_timer -lboost_serialization -lboost_system 
   ifeq ($(strip $(USE_MPI)), yes)   
      LFLAGS += -lboost_mpi
   endif
endif
LFLAGS += -L${HDF5}/lib -lhdf5
FLAGS = -I${BOOST}/include -I${HDF5}/include -I./extlibs/HighFive-master/include \
	${INCLUDE_DIR} -std=c++17
#FLAGS = -I${BOOST}/include -I${HDF5}/include -I./extlibs \
	${INCLUDE_DIR} -std=c++17

ifeq ($(strip $(USE_GCC)),yes)
   # GCC compiler
   ifeq ($(strip $(DEBUG)),yes)
      FLAGS += -DDEBUG -g -O0 -Wall 
   else
      FLAGS += -DNDEBUG -g -O2 -Wall 
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
      FLAGS += -DNDEBUG -g -O2 -Wall 
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
   # https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html	
   # parallel version of MKL
   MATH = -L$(MATHLIB) -Wl,-rpath,$(MATHLIB) \
          -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl \
   	  -liomp5
   # Use GNU OpenMP library: -lmkl_gnu_thread -lgomp replace -liomp5
endif
# quaternion matrix diagonalization
MATH += -L./extlibs/zquatev -lzquatev

LFLAGS += ${MATH}

SRC = src
BIN_DIR = ./bin
OBJ_DIR = ./obj

# all dependence
SRC_DIR_CORE = ./$(SRC)/core
SRC_DIR_IO   = ./$(SRC)/io
SRC_DIR_CI   = ./$(SRC)/ci
SRC_DIR_QT   = ./$(SRC)/ctns/qtensor
SRC_DIR_CTNS = ./$(SRC)/ctns
INCLUDE_DIR = -I$(SRC_DIR_CORE) \
	      -I$(SRC_DIR_IO) \
	      -I$(SRC_DIR_CI) \
	      -I$(SRC_DIR_QT) \
	      -I$(SRC_DIR_CTNS)
SRC_DEP = $(wildcard $(SRC_DIR_CORE)/*.cpp \
	  	     $(SRC_DIR_IO)/*.cpp  \
	  	     $(SRC_DIR_CI)/*.cpp \
	  	     $(SRC_DIR_QT)/*.cpp \
	  	     $(SRC_DIR_CTNS)/*.cpp)
OBJ_DEP = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_DEP}))

# all the files with main functions
SRC_ALL = $(SRC_DEP) 
SRC_ALL += $(wildcard ./$(SRC)/drivers/*.cpp)
OBJ_ALL = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir ${SRC_ALL}))

all: depend \
     $(BIN_DIR)/tests_core.x \
     $(BIN_DIR)/tests_ci.x \
     $(BIN_DIR)/tests_ctns.x \
     $(BIN_DIR)/sci.x \
     $(BIN_DIR)/ctns.x 

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
	mkdir -p $(BIN_DIR) $(OBJ_DIR); \
	echo $(SRC_ALL); \
	$(CXX) $(FLAGS) -I${BOOST}/include -MM $(SRC_ALL) > $$$$.depend; \
	sed 's,\([^.]*\.o\),$(OBJ_DIR)/\1,' < $$$$.depend > .depend; \
	echo 'finish dependency check!'; \
	rm -f $$$$.depend # $$$$ id number 
-include .depend

# Executables
$(BIN_DIR)/tests_core.x: $(OBJ_DIR)/tests_core.o $(OBJ_DEP)
	@echo "=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(BIN_DIR)/tests_ci.x: $(OBJ_DIR)/tests_ci.o $(OBJ_DEP)
	@echo "=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

$(BIN_DIR)/tests_ctns.x: $(OBJ_DIR)/tests_ctns.o $(OBJ_DEP)
	@echo "=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS) 

# Main: sci & ctns
$(BIN_DIR)/sci.x: $(OBJ_DIR)/sci.o $(OBJ_DEP)
	@echo "=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS) 

$(BIN_DIR)/ctns.x: $(OBJ_DIR)/ctns.o $(OBJ_DEP)
	@echo "=== LINK $@"
	@echo $(OBJ_DEP)
	$(CXX) $(FLAGS) -o $@ $^ $(LFLAGS)

# Needs to be here! 
$(OBJ_ALL):
	@echo "=== COMPILE $@ FROM $<" # just from *.cpp is sufficient	
	$(CXX) $(FLAGS) -o $@ -c $< 

clean:
	rm -f obj/*.o
	rm -f bin/*.x
	rm -f *.depend
