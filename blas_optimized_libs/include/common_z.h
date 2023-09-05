#ifndef __COMMON_Z_H__
#define __COMMON_Z_H__

#define ZGEMM_SMALL_KERNEL_NN   zgemm_small_kernel_nn
#define ZGEMM_SMALL_KERNEL_NT   zgemm_small_kernel_nt
#define ZGEMM_SMALL_KERNEL_NR   zgemm_small_kernel_nr
#define ZGEMM_SMALL_KERNEL_NC   zgemm_small_kernel_nc

#define ZGEMM_SMALL_KERNEL_TN   zgemm_small_kernel_tn
#define ZGEMM_SMALL_KERNEL_TT   zgemm_small_kernel_tt
#define ZGEMM_SMALL_KERNEL_TR   zgemm_small_kernel_tr
#define ZGEMM_SMALL_KERNEL_TC   zgemm_small_kernel_tc

#define ZGEMM_SMALL_KERNEL_RN   zgemm_small_kernel_rn
#define ZGEMM_SMALL_KERNEL_RT   zgemm_small_kernel_rt
#define ZGEMM_SMALL_KERNEL_RR   zgemm_small_kernel_rr
#define ZGEMM_SMALL_KERNEL_RC   zgemm_small_kernel_rc

#define ZGEMM_SMALL_KERNEL_CN   zgemm_small_kernel_cn
#define ZGEMM_SMALL_KERNEL_CT   zgemm_small_kernel_ct
#define ZGEMM_SMALL_KERNEL_CR   zgemm_small_kernel_cr
#define ZGEMM_SMALL_KERNEL_CC   zgemm_small_kernel_cc

#define ZGEMM_SMALL_KERNEL_B0_NN   zgemm_small_kernel_b0_nn
#define ZGEMM_SMALL_KERNEL_B0_NT   zgemm_small_kernel_b0_nt
#define ZGEMM_SMALL_KERNEL_B0_NR   zgemm_small_kernel_b0_nr
#define ZGEMM_SMALL_KERNEL_B0_NC   zgemm_small_kernel_b0_nc

#define ZGEMM_SMALL_KERNEL_B0_TN   zgemm_small_kernel_b0_tn
#define ZGEMM_SMALL_KERNEL_B0_TT   zgemm_small_kernel_b0_tt
#define ZGEMM_SMALL_KERNEL_B0_TR   zgemm_small_kernel_b0_tr
#define ZGEMM_SMALL_KERNEL_B0_TC   zgemm_small_kernel_b0_tc

#define ZGEMM_SMALL_KERNEL_B0_RN   zgemm_small_kernel_b0_rn
#define ZGEMM_SMALL_KERNEL_B0_RT   zgemm_small_kernel_b0_rt
#define ZGEMM_SMALL_KERNEL_B0_RR   zgemm_small_kernel_b0_rr
#define ZGEMM_SMALL_KERNEL_B0_RC   zgemm_small_kernel_b0_rc

#define ZGEMM_SMALL_KERNEL_B0_CN   zgemm_small_kernel_b0_cn
#define ZGEMM_SMALL_KERNEL_B0_CT   zgemm_small_kernel_b0_ct
#define ZGEMM_SMALL_KERNEL_B0_CR   zgemm_small_kernel_b0_cr
#define ZGEMM_SMALL_KERNEL_B0_CC   zgemm_small_kernel_b0_cc


#define ZGEMM_ONCOPY    zgemm_oncopy
#define ZGEMM_OTCOPY    zgemm_otcopy

#if ZGEMM_DEFAULT_UNROLL_M == ZGEMM_DEFAULT_UNROLL_N
#define ZGEMM_INCOPY    zgemm_oncopy
#define ZGEMM_ITCOPY    zgemm_otcopy
#else
#define ZGEMM_INCOPY    zgemm_incopy
#define ZGEMM_ITCOPY    zgemm_itcopy
#endif

#define ZGEMM_BETA    zgemm_beta

#define ZGEMM_KERNEL_N    zgemm_kernel_n
#define ZGEMM_KERNEL_L    zgemm_kernel_l
#define ZGEMM_KERNEL_R    zgemm_kernel_r
#define ZGEMM_KERNEL_B    zgemm_kernel_b

#define ZGEMM_NN    zgemm_nn
#define ZGEMM_CN    zgemm_cn
#define ZGEMM_TN    zgemm_tn
#define ZGEMM_NC    zgemm_nc
#define ZGEMM_NT    zgemm_nt
#define ZGEMM_CC    zgemm_cc
#define ZGEMM_CT    zgemm_ct
#define ZGEMM_TC    zgemm_tc
#define ZGEMM_TT    zgemm_tt
#define ZGEMM_NR    zgemm_nr
#define ZGEMM_TR    zgemm_tr
#define ZGEMM_CR    zgemm_cr
#define ZGEMM_RN    zgemm_rn
#define ZGEMM_RT    zgemm_rt
#define ZGEMM_RC    zgemm_rc
#define ZGEMM_RR    zgemm_rr


#define	ZCOPY_K			zcopy_k
#define	ZNRM2_K			znrm2_k


#endif
