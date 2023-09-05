#ifndef __COMMON_C_H__
#define __COMMON_C_H__

#define CGEMM_SMALL_KERNEL_NN   cgemm_small_kernel_nn
#define CGEMM_SMALL_KERNEL_NT   cgemm_small_kernel_nt
#define CGEMM_SMALL_KERNEL_NR   cgemm_small_kernel_nr
#define CGEMM_SMALL_KERNEL_NC   cgemm_small_kernel_nc

#define CGEMM_SMALL_KERNEL_TN   cgemm_small_kernel_tn
#define CGEMM_SMALL_KERNEL_TT   cgemm_small_kernel_tt
#define CGEMM_SMALL_KERNEL_TR   cgemm_small_kernel_tr
#define CGEMM_SMALL_KERNEL_TC   cgemm_small_kernel_tc

#define CGEMM_SMALL_KERNEL_RN   cgemm_small_kernel_rn
#define CGEMM_SMALL_KERNEL_RT   cgemm_small_kernel_rt
#define CGEMM_SMALL_KERNEL_RR   cgemm_small_kernel_rr
#define CGEMM_SMALL_KERNEL_RC   cgemm_small_kernel_rc

#define CGEMM_SMALL_KERNEL_CN   cgemm_small_kernel_cn
#define CGEMM_SMALL_KERNEL_CT   cgemm_small_kernel_ct
#define CGEMM_SMALL_KERNEL_CR   cgemm_small_kernel_cr
#define CGEMM_SMALL_KERNEL_CC   cgemm_small_kernel_cc

#define CGEMM_SMALL_KERNEL_B0_NN   cgemm_small_kernel_b0_nn
#define CGEMM_SMALL_KERNEL_B0_NT   cgemm_small_kernel_b0_nt
#define CGEMM_SMALL_KERNEL_B0_NR   cgemm_small_kernel_b0_nr
#define CGEMM_SMALL_KERNEL_B0_NC   cgemm_small_kernel_b0_nc

#define CGEMM_SMALL_KERNEL_B0_TN   cgemm_small_kernel_b0_tn
#define CGEMM_SMALL_KERNEL_B0_TT   cgemm_small_kernel_b0_tt
#define CGEMM_SMALL_KERNEL_B0_TR   cgemm_small_kernel_b0_tr
#define CGEMM_SMALL_KERNEL_B0_TC   cgemm_small_kernel_b0_tc

#define CGEMM_SMALL_KERNEL_B0_RN   cgemm_small_kernel_b0_rn
#define CGEMM_SMALL_KERNEL_B0_RT   cgemm_small_kernel_b0_rt
#define CGEMM_SMALL_KERNEL_B0_RR   cgemm_small_kernel_b0_rr
#define CGEMM_SMALL_KERNEL_B0_RC   cgemm_small_kernel_b0_rc

#define CGEMM_SMALL_KERNEL_B0_CN   cgemm_small_kernel_b0_cn
#define CGEMM_SMALL_KERNEL_B0_CT   cgemm_small_kernel_b0_ct
#define CGEMM_SMALL_KERNEL_B0_CR   cgemm_small_kernel_b0_cr
#define CGEMM_SMALL_KERNEL_B0_CC   cgemm_small_kernel_b0_cc


#define CGEMM_ONCOPY    cgemm_oncopy
#define CGEMM_OTCOPY    cgemm_otcopy

#if CGEMM_DEFAULT_UNROLL_M == CGEMM_DEFAULT_UNROLL_N
#define CGEMM_INCOPY    cgemm_oncopy
#define CGEMM_ITCOPY    cgemm_otcopy
#else
#define CGEMM_INCOPY    cgemm_incopy
#define CGEMM_ITCOPY    cgemm_itcopy
#endif

#define CGEMM_BETA    cgemm_beta

#define CGEMM_KERNEL_N    cgemm_kernel_n
#define CGEMM_KERNEL_L    cgemm_kernel_l
#define CGEMM_KERNEL_R    cgemm_kernel_r
#define CGEMM_KERNEL_B    cgemm_kernel_b

#define CGEMM_NN    cgemm_nn
#define CGEMM_CN    cgemm_cn
#define CGEMM_TN    cgemm_tn
#define CGEMM_NC    cgemm_nc
#define CGEMM_NT    cgemm_nt
#define CGEMM_CC    cgemm_cc
#define CGEMM_CT    cgemm_ct
#define CGEMM_TC    cgemm_tc
#define CGEMM_TT    cgemm_tt
#define CGEMM_NR    cgemm_nr
#define CGEMM_TR    cgemm_tr
#define CGEMM_CR    cgemm_cr
#define CGEMM_RN    cgemm_rn
#define CGEMM_RT    cgemm_rt
#define CGEMM_RC    cgemm_rc
#define CGEMM_RR    cgemm_rr

#define	CCOPY_K			ccopy_k
#define	CNRM2_K			cnrm2_k

#endif
