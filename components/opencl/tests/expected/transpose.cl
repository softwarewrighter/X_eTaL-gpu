// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// transpose0: %t := o_\, the axes reversed, [2 3] to [3 2]; a gather, one work-item per result item.
__kernel void transpose0(__global const long* b_m, __global long* b_t) {
    long k = get_global_id(0);
    if (k >= 6) return;
    long r = k, src = 0;
    src += (r % 2) * 3; r /= 2;
    src += r * 1;
    b_t[k] = b_m[src];
}

// transpose1: %t_3 := o_\, the axes reversed, [2 3 4] to [4 3 2]; a gather, one work-item per result item.
__kernel void transpose1(__global const long* b_t_2, __global long* b_t_3) {
    long k = get_global_id(0);
    if (k >= 24) return;
    long r = k, src = 0;
    src += (r % 2) * 12; r /= 2;
    src += (r % 3) * 4; r /= 3;
    src += r * 1;
    b_t_3[k] = b_t_2[src];
}

// transpose2: %tr := 3 1 2 t_ranspose, [2 3 4] to [3 4 2]; a gather, one work-item per result item.
__kernel void transpose2(__global const long* b_t_2, __global long* b_tr) {
    long k = get_global_id(0);
    if (k >= 24) return;
    long r = k, src = 0;
    src += (r % 2) * 12; r /= 2;
    src += (r % 4) * 1; r /= 4;
    src += r * 4;
    b_tr[k] = b_t_2[src];
}

// transpose3: %t_4 := 1 3 2 t_ranspose, [2 3 4] to [2 4 3]; a gather, one work-item per result item.
__kernel void transpose3(__global const long* b_t_2, __global long* b_t_4) {
    long k = get_global_id(0);
    if (k >= 24) return;
    long r = k, src = 0;
    src += (r % 3) * 4; r /= 3;
    src += (r % 4) * 1; r /= 4;
    src += r * 12;
    b_t_4[k] = b_t_2[src];
}

// transpose4: %t_6 := o_\, the axes reversed, [4 3] to [3 4]; a gather, one work-item per result item.
__kernel void transpose4(__global const float* b_k, __global float* b_t_6) {
    long k = get_global_id(0);
    if (k >= 12) return;
    long r = k, src = 0;
    src += (r % 4) * 3; r /= 4;
    src += r * 1;
    b_t_6[k] = b_k[src];
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

// transpose6: %t_7 := o_\, the axes reversed, [4 2] to [2 4]; a gather, one work-item per result item.
__kernel void transpose6(__global const float* b_x_2, __global float* b_t_7) {
    long k = get_global_id(0);
    if (k >= 8) return;
    long r = k, src = 0;
    src += (r % 4) * 2; r /= 4;
    src += r * 1;
    b_t_7[k] = b_x_2[src];
}
