#ifndef __COMMON_LEVEL1_H__
#define __COMMON_LEVEL1_H__
int    scopy_k(BLASLONG, float  *, BLASLONG, float  *, BLASLONG);
int    dcopy_k(BLASLONG, double *, BLASLONG, double *, BLASLONG);
int    ccopy_k(BLASLONG, float  *, BLASLONG, float  *, BLASLONG);
int    zcopy_k(BLASLONG, double *, BLASLONG, double *, BLASLONG);

float   snrm2_k(BLASLONG, float  *, BLASLONG);
double  dnrm2_k(BLASLONG, double *, BLASLONG);
float   cnrm2_k(BLASLONG, float  *, BLASLONG);
double  znrm2_k(BLASLONG, double *, BLASLONG);
#endif
