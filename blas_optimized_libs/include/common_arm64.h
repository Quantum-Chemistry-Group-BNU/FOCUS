#ifndef __COMMON_ARM64_H_
#define __COMMON_ARM64_H_


#if defined(ASSEMBLER)
#define REALNAME CNAME	
#define PROLOGUE \
	.text; \
  .align 5 ;\
	.globl REALNAME;\
	.type REALNAME, %function;\
REALNAME:;

#define EPILOGUE  

//#define REGISTERSAVE \
//	add	sp,  sp,      #-(9 * 16) ;\
//	stp	d8, d9,   [sp, #(0 * 16)];\
//	stp	d10, d11, [sp, #(1 * 16)];\
//	stp	d12, d13, [sp, #(2 * 16)];\
//	stp	d14, d15, [sp, #(3 * 16)];\
//	stp	x19, x20, [sp, #(4 * 16)];\
//	stp	x21, x22, [sp, #(5 * 16)];\
//	stp	x23, x24, [sp, #(6 * 16)];\
//	stp	x25, x26, [sp, #(7 * 16)];\
//	stp	x27, x28, [sp, #(8 * 16)];
//
//// set return value
//#define REGISTERREVERT \
//	ldp	d8, d9,   [sp, #(0 * 16)];\
//	ldp	d10, d11, [sp, #(1 * 16)];\
//	ldp	d12, d13, [sp, #(2 * 16)];\
//	ldp	d14, d15, [sp, #(3 * 16)];\
//	ldp	x19, x20, [sp, #(4 * 16)];\
//	ldp	x21, x22, [sp, #(5 * 16)];\
//	ldp	x23, x24, [sp, #(6 * 16)];\
//	ldp	x25, x26, [sp, #(7 * 16)];\
//	ldp	x27, x28, [sp, #(8 * 16)];\
//	add	sp,          sp, #(9*16);

#endif//ASSEMBLER

#ifdef C_MSVC
#include <intrin.h>
#define MB __dmb(_ARM64_BARRIER_ISH)
#define WMB __dmb(_ARM64_BARRIER_ISHST)
#define RMB __dmb(_ARM64_BARRIER_ISHLD)
#else
#define MB   __asm__ __volatile__ ("dmb  ish" : : : "memory")
#define WMB  __asm__ __volatile__ ("dmb  ishst" : : : "memory")
#define RMB  __asm__ __volatile__ ("dmb  ishld" : : : "memory")
#endif


#define SEEK_ADDRESS

#ifndef PAGESIZE
#define PAGESIZE        ( 4 << 10)
#endif
#define HUGE_PAGESIZE   ( 4 << 20)

#ifndef BUFFERSIZE
#define BUFFER_SIZE     (32 << 20)
#else
#define BUFFER_SIZE (32 << BUFFERSIZE)
#endif

#define BASE_ADDRESS (START_ADDRESS - BUFFER_SIZE * MAX_CPU_NUMBER)


#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#endif
