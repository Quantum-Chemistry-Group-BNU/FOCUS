#ifndef __COMMON_H__
#define __COMMON_H__

#include "param.h"
#include "common_macro.h"
#include "common_param.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))

#if defined(DOUBLE)
#define FLOAT double
#define SIZE  8
#define  BASE_SHIFT 3
#define ZBASE_SHIFT 4
#else
#define FLOAT float
#define SIZE    4
#define  BASE_SHIFT 2
#define ZBASE_SHIFT 3
#endif
#ifndef XFLOAT
#define XFLOAT  FLOAT
#endif

#ifndef IFLOAT
#define IFLOAT  FLOAT
#endif

#ifndef COMPLEX
#define COMPSIZE  1
#else
#define COMPSIZE  2
#endif

#ifndef ZERO
#ifdef XDOUBLE
#define ZERO  0.e0L
#elif defined DOUBLE
#define ZERO  0.e0
#else
#define ZERO  0.e0f
#endif
#endif

#ifndef ONE
#ifdef XDOUBLE
#define ONE  1.e0L
#elif defined DOUBLE
#define ONE  1.e0
#else
#define ONE  1.e0f
#endif
#endif

#include "common_arm64.h"

#ifndef ASSEMBLER

#include <sys/mman.h>
#include <sys/shm.h>
#include <time.h>
#include <math.h>

#ifndef MAX_PARALLEL_NUMBER
#define MAX_PARALLEL_NUMBER 1
#endif

#ifndef MAX_CPU_NUMBER
#define MAX_CPU_NUMBER 2
#endif

#define NUM_BUFFERS MAX(50,(MAX_CPU_NUMBER * 2 * MAX_PARALLEL_NUMBER))

#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#ifdef __NetBSD__
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANON)
#else
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)
#endif

typedef char* env_var_t;
#define readenv(p, n) ((p)=getenv(n))



#if   defined(USE_PTHREAD_LOCK)
#define   LOCK_COMMAND(x)   pthread_mutex_lock(x)
#define UNLOCK_COMMAND(x)   pthread_mutex_unlock(x)
#elif defined(USE_PTHREAD_SPINLOCK)
#ifndef ASSEMBLER
typedef volatile int pthread_spinlock_t;
int pthread_spin_lock (pthread_spinlock_t *__lock);
int pthread_spin_unlock (pthread_spinlock_t *__lock);
#endif
#define   LOCK_COMMAND(x)   pthread_spin_lock(x)
#define UNLOCK_COMMAND(x)   pthread_spin_unlock(x)
#else
#define   LOCK_COMMAND(x)   blas_lock(x)
#define UNLOCK_COMMAND(x)   blas_unlock(x)
#endif






typedef long BLASLONG;
typedef unsigned long BLASULONG;
#ifdef USE64BITINT
typedef BLASLONG blasint;
#else
typedef int blasint;
#endif

typedef struct {
  void *a, *b, *c, *d, *alpha, *beta;
  BLASLONG	m, n, k, lda, ldb, ldc, ldd;

#ifdef SMP
  void *common;
  BLASLONG nthreads;
#endif

#ifdef PARAMTEST
  BLASLONG gemm_p, gemm_q, gemm_r;
#endif

#ifdef PREFETCHTEST
  BLASLONG prea, preb, prec, pred;
#endif

} blas_arg_t;



#define YIELDING        __asm__ __volatile__ ("nop;nop;nop;nop;nop;nop;nop;nop; \n");

#if !defined(BLAS_LOCK_DEFINED) && defined(__GNUC__)
static void __inline blas_lock(volatile BLASULONG *address){

  do {
    while (*address) {YIELDING;};

  } while (!__sync_bool_compare_and_swap(address, 0, 1));
}
#define BLAS_LOCK_DEFINED
#endif

#include "common_level1.h"
#include "common_level3.h"

#include <time.h>
#include <sys/time.h>


#ifdef TIMING
#define START_RPCC()		rpcc_counter = rpcc()
#define STOP_RPCC(COUNTER)	COUNTER  += rpcc() - rpcc_counter
#else
#define START_RPCC()
#define STOP_RPCC(COUNTER)
#endif

static inline unsigned long long rpcc(void){
#ifdef USE_MONOTONIC
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (unsigned long long)ts.tv_sec * 1000000000ull + ts.tv_nsec;
#else
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (unsigned long long)tv.tv_sec * 1000000000ull + tv.tv_usec * 1000;
#endif
}

/* Common Memory Management Routine */
void  blas_set_parameter(void);
int   blas_get_cpu_number(void);
void *blas_memory_alloc  (int);
void  blas_memory_free   (void *);
void *blas_memory_alloc_nolock  (int); //use malloc without blas_lock
void  blas_memory_free_nolock   (void *);

int  get_num_procs (void);

#if defined(OS_LINUX) && defined(SMP) && !defined(NO_AFFINITY)
int  get_num_nodes (void);
int get_num_proc   (int);
int get_node_equal (void);
#endif

void goto_set_num_threads(int);

void gotoblas_affinity_init(void);
void gotoblas_affinity_quit(void);
void gotoblas_dynamic_init(void);
void gotoblas_dynamic_quit(void);
void gotoblas_profile_init(void);
void gotoblas_profile_quit(void);
	
int support_avx512(void);	

#ifdef USE_OPENMP

#ifndef C_MSVC
int omp_in_parallel(void);
int omp_get_num_procs(void);
#else
__declspec(dllimport) int __cdecl omp_in_parallel(void);
__declspec(dllimport) int __cdecl omp_get_num_procs(void);
#endif

#ifdef HAVE_C11
#if defined(C_GCC) && ( __GNUC__ < 7) 
// workaround for GCC bug 65467
#ifndef _Atomic
#define _Atomic volatile
#endif
#endif
#include <stdatomic.h>
#else
#ifndef _Atomic
#define _Atomic volatile
#endif
#endif

#else
#ifdef __ELF__
int omp_in_parallel  (void) __attribute__ ((weak));
int omp_get_num_procs(void) __attribute__ ((weak));
#endif
#endif

static __inline void blas_unlock(volatile BLASULONG *address){
  MB;
  *address = 0;
}

#ifdef OS_WINDOWSSTORE
static __inline int readenv_atoi(char *env) {
	return 0;
}
#else
#ifdef OS_WINDOWS
static __inline int readenv_atoi(char *env) {
  env_var_t p;
  return readenv(p,env) ? 0 : atoi(p);
}
#else
//static __inline int readenv_atoi(char *env) {
//  char *p;
//  if (( p = getenv(env) ))
//  	return (atoi(p));
//  else
//	return(0);
//}
#endif
#endif
#endif//ASSEMBLER


#endif
