#ifndef __COMMON_D_H__
#define __COMMON_D_H__

#define DGEMM_SMALL_KERNEL_NN   dgemm_small_kernel_nn
#define DGEMM_SMALL_KERNEL_NT   dgemm_small_kernel_nt
#define DGEMM_SMALL_KERNEL_TN   dgemm_small_kernel_tn
#define DGEMM_SMALL_KERNEL_TT   dgemm_small_kernel_tt

#define DGEMM_SMALL_KERNEL_B0_NN   dgemm_small_kernel_b0_nn
#define DGEMM_SMALL_KERNEL_B0_NT   dgemm_small_kernel_b0_nt
#define DGEMM_SMALL_KERNEL_B0_TN   dgemm_small_kernel_b0_tn
#define DGEMM_SMALL_KERNEL_B0_TT   dgemm_small_kernel_b0_tt


#define DGEMM_ONCOPY    dgemm_oncopy
#define DGEMM_OTCOPY    dgemm_otcopy

#if DGEMM_DEFAULT_UNROLL_M == DGEMM_DEFAULT_UNROLL_N
#define DGEMM_INCOPY    dgemm_oncopy
#define DGEMM_ITCOPY    dgemm_otcopy
#else
#define DGEMM_INCOPY    dgemm_incopy
#define DGEMM_ITCOPY    dgemm_itcopy
#endif

#define DGEMM_BETA    dgemm_beta
#define DGEMM_KERNEL    dgemm_kernel

#define DGEMM_NN    dgemm_nn
#define DGEMM_CN    dgemm_tn
#define DGEMM_TN    dgemm_tn
#define DGEMM_NC    dgemm_nt
#define DGEMM_NT    dgemm_nt
#define DGEMM_CC    dgemm_tt
#define DGEMM_CT    dgemm_tt
#define DGEMM_TC    dgemm_tt
#define DGEMM_TT    dgemm_tt
#define DGEMM_NR    dgemm_nn
#define DGEMM_TR    dgemm_tn
#define DGEMM_CR    dgemm_tn
#define DGEMM_RN    dgemm_nn
#define DGEMM_RT    dgemm_nt
#define DGEMM_RC    dgemm_nt
#define DGEMM_RR    dgemm_nn


#define	DCOPY_K			dcopy_k
#define	DNRM2_K			dnrm2_k

#endif
