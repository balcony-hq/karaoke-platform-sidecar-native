#include "native_simd.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

int vocalarc_simd_available(void) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4] = {0, 0, 0, 0};
    __cpuid(registers, 0);
    int maximum_leaf = registers[0];
    if (maximum_leaf < 1) return 0;

    __cpuidex(registers, 1, 0);
    const int required_ecx = (1 << 12) | (1 << 27) | (1 << 28); /* FMA, OSXSAVE, AVX */
    if ((registers[2] & required_ecx) != required_ecx) return 0;
    if ((_xgetbv(0) & 0x6) != 0x6) return 0; /* XMM/YMM state enabled by the OS */
    if (maximum_leaf < 7) return 0;

    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0; /* AVX2 */
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}
