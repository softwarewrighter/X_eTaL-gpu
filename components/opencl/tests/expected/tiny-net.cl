// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// slice_float: a take or a drop along an axis: of every line (outer lines
// of len items, inner apart), the keep items from start; one
// work-item per item kept.
__kernel void slice_float(__global const float* in, __global float* out, const long outer, const long len, const long inner, const long start, const long keep) {
    long k = get_global_id(0);
    if (k >= outer * keep * inner) return;
    long o = k / (keep * inner), r = k % (keep * inner);
    long j = r / inner, i = r % inner;
    out[k] = in[(o * len + start + j) * inner + i];
}

// table1: %tb := a 'right t_able b, 4 by 5 items, one work-item each.
__kernel void table1(__global const long* b_a, __global const float* b_take, __global float* b_tb) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb[k] = b_take[k % nb];
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

// k1: 20 elements, one work-item each; computes %add %max
__kernel void k1(__global const float* b_mm, __global const float* b_tb, __global float* b_max) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    float t_add = (b_mm[i] + b_tb[i]);
    float t_max = fmax(t_add, 0.0f);
    b_max[i] = t_max;
}

// table6: %tb_2 := a 'right t_able b, 4 by 2 items, one work-item each.
__kernel void table6(__global const long* b_a_2, __global const float* b_take_2, __global float* b_tb_2) {
    long k = get_global_id(0);
    const long nb = 2;
    if (k >= 8) return;
    b_tb_2[k] = b_take_2[k % nb];
}

// k3: 8 elements, one work-item each; computes %add_2
__kernel void k3(__global const float* b_mm_2, __global const float* b_tb_2, __global float* b_add_2) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_add_2 = (b_mm_2[i] + b_tb_2[i]);
    b_add_2[i] = t_add_2;
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

// table11: %tb_3 := a 'left t_able b, 4 by 2 items, one work-item each.
__kernel void table11(__global const float* b_maxr, __global const long* b_b, __global float* b_tb_3) {
    long k = get_global_id(0);
    const long nb = 2;
    if (k >= 8) return;
    b_tb_3[k] = b_maxr[k / nb];
}

// k5: 8 elements, one work-item each; computes %sub %exp
__kernel void k5(__global const float* b_add_2, __global const float* b_tb_3, __global float* b_exp) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_sub = (b_add_2[i] - b_tb_3[i]);
    float t_exp = exp(t_sub);
    b_exp[i] = t_exp;
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

// table14: %tb_4 := a 'left t_able b, 4 by 2 items, one work-item each.
__kernel void table14(__global const float* b_addr, __global const long* b_b_2, __global float* b_tb_4) {
    long k = get_global_id(0);
    const long nb = 2;
    if (k >= 8) return;
    b_tb_4[k] = b_addr[k / nb];
}

// k7: 8 elements, one work-item each; computes %div
__kernel void k7(__global const float* b_exp, __global const float* b_tb_4, __global float* b_div) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_div = (b_exp[i] / b_tb_4[i]);
    b_div[i] = t_div;
}
