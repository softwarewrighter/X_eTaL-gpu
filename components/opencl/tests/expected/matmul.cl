// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// matmul_long: a by b, a's last axis (n) with b's first; one work-item per
// result item (i, j), its n products summed from the last to the first,
// as X_eTaL's '+ '* i_nner sums them.
__kernel void matmul_long(__global const long* a, __global const long* b, __global long* c, const long m, const long n, const long p) {
    long ij = get_global_id(0);
    if (ij >= m * p) return;
    long i = ij / p, j = ij % p;
    long acc = a[i * n + n - 1] * b[(n - 1) * p + j];
    for (long k = n - 2; k >= 0; k--) acc = a[i * n + k] * b[k * p + j] + acc;
    c[ij] = acc;
}

// k0: 12 elements, one work-item each; computes %af %f
__kernel void k0(__global const long* b_a, __global float* b_f) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    float t_af = (float)b_a[i];
    float t_f = (t_af / 4.0f);
    b_f[i] = t_f;
}

// k1: 12 elements, one work-item each; computes %g
__kernel void k1(__global const float* b_g0, __global float* b_g) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    float t_g = (b_g0[i] / 8.0f);
    b_g[i] = t_g;
}

// matmul_float: a by b, a's last axis (n) with b's first; one work-item per
// result item (i, j), its n products summed from the last to the first,
// as X_eTaL's '+ '* i_nner sums them.
__kernel void matmul_float(__global const float* a, __global const float* b, __global float* c, const long m, const long n, const long p) {
    long ij = get_global_id(0);
    if (ij >= m * p) return;
    long i = ij / p, j = ij % p;
    float acc = a[i * n + n - 1] * b[(n - 1) * p + j];
    for (long k = n - 2; k >= 0; k--) acc = a[i * n + k] * b[k * p + j] + acc;
    c[ij] = acc;
}
