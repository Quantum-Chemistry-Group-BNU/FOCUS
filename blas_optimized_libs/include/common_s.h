#ifndef __COMMON_S_H__
#define __COMMON_S_H__

#define SGEMM_SMALL_KERNEL_NN   sgemm_small_kernel_nn
#define SGEMM_SMALL_KERNEL_NT   sgemm_small_kernel_nt
#define SGEMM_SMALL_KERNEL_TN   sgemm_small_kernel_tn
#define SGEMM_SMALL_KERNEL_TT   sgemm_small_kernel_tt

#define SGEMM_SMALL_KERNEL_B0_NN   sgemm_small_kernel_b0_nn
#define SGEMM_SMALL_KERNEL_B0_NT   sgemm_small_kernel_b0_nt
#define SGEMM_SMALL_KERNEL_B0_TN   sgemm_small_kernel_b0_tn
#define SGEMM_SMALL_KERNEL_B0_TT   sgemm_small_kernel_b0_tt

#define SGEMM_ONCOPY    sgemm_oncopy
#define SGEMM_OTCOPY    sgemm_otcopy

#if SGEMM_DEFAULT_UNROLL_M == SGEMM_DEFAULT_UNROLL_N
#define SGEMM_INCOPY    sgemm_oncopy
#define SGEMM_ITCOPY    sgemm_otcopy
#else
#define SGEMM_INCOPY    sgemm_incopy
#define SGEMM_ITCOPY    sgemm_itcopy
#endif

#define SGEMM_BETA    sgemm_beta
#define SGEMM_KERNEL    sgemm_kernel

#define SGEMM_NN    sgemm_nn
#define SGEMM_CN    sgemm_tn
#define SGEMM_TN    sgemm_tn
#define SGEMM_NC    sgemm_nt
#define SGEMM_NT    sgemm_nt
#define SGEMM_CC    sgemm_tt
#define SGEMM_CT    sgemm_tt
#define SGEMM_TC    sgemm_tt
#define SGEMM_TT    sgemm_tt
#define SGEMM_NR    sgemm_nn
#define SGEMM_TR    sgemm_tn
#define SGEMM_CR    sgemm_tn
#define SGEMM_RN    sgemm_nn
#define SGEMM_RT    sgemm_nt
#define SGEMM_RC    sgemm_nt
#define SGEMM_RR    sgemm_nn


#define	SCOPY_K			scopy_k
#define	SNRM2_K			snrm2_k
#endif
