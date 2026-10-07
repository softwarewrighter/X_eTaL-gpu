// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// copy_float: count items from offset, one work-item each (a take or a
// drop along the first axis keeps contiguous rows).
__kernel void copy_float(__global const float* in, __global float* out, const long offset, const long count) {
    long k = get_global_id(0);
    if (k < count) out[k] = in[offset + k];
}

// table2: %b1 := a 'right t_able b, 4 by 5 items, one work-item each.
__kernel void table2(__global const long* b_o4, __global const float* b_w1b, __global float* b_b1) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_b1[k] = b_w1b[k % nb];
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

// k1: 20 elements, one work-item each; computes %z1 %h
__kernel void k1(__global const float* b_xw1, __global const float* b_b1, __global float* b_h) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    float t_z1 = (b_xw1[i] + b_b1[i]);
    float t_h = fmax(t_z1, 0.0f);
    b_h[i] = t_h;
}

// table7: %b2 := a 'right t_able b, 4 by 2 items, one work-item each.
__kernel void table7(__global const long* b_o4, __global const float* b_w2b, __global float* b_b2) {
    long k = get_global_id(0);
    const long nb = 2;
    if (k >= 8) return;
    b_b2[k] = b_w2b[k % nb];
}

// k3: 8 elements, one work-item each; computes %z2
__kernel void k3(__global const float* b_hw2, __global const float* b_b2, __global float* b_z2) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_z2 = (b_hw2[i] + b_b2[i]);
    b_z2[i] = t_z2;
}

// reduce_axis_max_float: along one axis, one work-item per result item; its line of
// len items (stride inner) folded from the right, as X_eTaL's r_/ does.
__kernel void reduce_axis_max_float(__global const float* in, __global float* out, const long outer, const long len, const long inner) {
    long k = get_global_id(0);
    if (k >= outer * inner) return;
    long o = k / inner, i = k % inner;
    long base = o * len * inner + i;
    float acc = in[base + (len - 1) * inner];
    for (long j = len - 2; j >= 0; j--) acc = fmax(in[base + j * inner], acc);
    out[k] = acc;
}

// table11: %mxs := a 'left t_able b, 4 by 2 items, one work-item each.
__kernel void table11(__global const float* b_mx, __global const long* b_c, __global float* b_mxs) {
    long k = get_global_id(0);
    const long nb = 2;
    if (k >= 8) return;
    b_mxs[k] = b_mx[k / nb];
}

// k5: 8 elements, one work-item each; computes %d %e
__kernel void k5(__global const float* b_z2, __global const float* b_mxs, __global float* b_e) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_d = (b_z2[i] - b_mxs[i]);
    float t_e = exp(t_d);
    b_e[i] = t_e;
}

// reduce_axis_add_float: along one axis, one work-item per result item; its line of
// len items (stride inner) folded from the right, as X_eTaL's r_/ does.
__kernel void reduce_axis_add_float(__global const float* in, __global float* out, const long outer, const long len, const long inner) {
    long k = get_global_id(0);
    if (k >= outer * inner) return;
    long o = k / inner, i = k % inner;
    long base = o * len * inner + i;
    float acc = in[base + (len - 1) * inner];
    for (long j = len - 2; j >= 0; j--) acc = in[base + j * inner] + acc;
    out[k] = acc;
}

// table14: %ss := a 'left t_able b, 4 by 2 items, one work-item each.
__kernel void table14(__global const float* b_s, __global const long* b_c, __global float* b_ss) {
    long k = get_global_id(0);
    const long nb = 2;
    if (k >= 8) return;
    b_ss[k] = b_s[k / nb];
}

// k7: 8 elements, one work-item each; computes %p
__kernel void k7(__global const float* b_e, __global const float* b_ss, __global float* b_p) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_p = (b_e[i] / b_ss[i]);
    b_p[i] = t_p;
}
