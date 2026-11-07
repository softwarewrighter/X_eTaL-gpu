// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

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

// table1: %tb := a 'left t_able b, 3 by 4 items, one work-item each.
__kernel void table1(__global const float* b_maxr, __global const long* b_b, __global float* b_tb) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_tb[k] = b_maxr[k / nb];
}

// k1: 12 elements, one work-item each; computes %eq %f %sub %mul
__kernel void k1(__global const float* b_m, __global const float* b_tb, __global float* b_f, __global float* b_mul) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    int t_eq = (b_m[i] == b_tb[i]);
    float t_f = (float)t_eq;
    float t_sub = (1.0f - t_f);
    float t_mul = (t_sub * 5.0f);
    b_f[i] = t_f;
    b_mul[i] = t_mul;
}

// table3: %tb_2 := a 'right t_able b, 3 by 4 items, one work-item each.
__kernel void table3(__global const long* b_a, __global const float* b_b_2, __global float* b_tb_2) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_tb_2[k] = b_b_2[k % nb];
}

// k3: 12 elements, one work-item each; computes %mul_2 %add
__kernel void k3(__global const float* b_f, __global const float* b_tb_2, __global const float* b_mul, __global float* b_add) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    float t_mul_2 = (b_f[i] * b_tb_2[i]);
    float t_add = (t_mul_2 + b_mul[i]);
    b_add[i] = t_add;
}

// reduce_axis_min_float: along one axis, one work-item per result item; its line of
// len items (stride inner) folded from the right, as X_eTaL's r_/ does.
__kernel void reduce_axis_min_float(__global const float* in, __global float* out, const long outer, const long len, const long inner) {
    long k = get_global_id(0);
    if (k >= outer * inner) return;
    long o = k / inner, i = k % inner;
    long base = o * len * inner + i;
    float acc = in[base + (len - 1) * inner];
    for (long j = len - 2; j >= 0; j--) acc = fmin(in[base + j * inner], acc);
    out[k] = acc;
}

// k4: 3 elements, one work-item each; computes %fl
__kernel void k4(__global const float* b_minr, __global long* b_fl) {
    size_t i = get_global_id(0);
    if (i >= 3) return;
    long t_fl = (long)floor(b_minr[i]);
    b_fl[i] = t_fl;
}

// k5: 3 elements, one work-item each; computes %sub_2 %add_2
__kernel void k5(__global const float* b_maxr, __global const float* b_minr_2, __global float* b_add_2) {
    size_t i = get_global_id(0);
    if (i >= 3) return;
    float t_sub_2 = (b_maxr[i] - b_minr_2[i]);
    float t_add_2 = (1.0f + t_sub_2);
    b_add_2[i] = t_add_2;
}

// table9: %tb_3 := a 'left t_able b, 3 by 4 items, one work-item each.
__kernel void table9(__global const float* b_add_2, __global const long* b_b, __global float* b_tb_3) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_tb_3[k] = b_add_2[k / nb];
}

// k7: 12 elements, one work-item each; computes %mul_3 %sub_3
__kernel void k7(__global const float* b_tb_3, __global const float* b_f, __global const float* b_m, __global float* b_sub_3) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    float t_mul_3 = (b_tb_3[i] * b_f[i]);
    float t_sub_3 = (b_m[i] - t_mul_3);
    b_sub_3[i] = t_sub_3;
}

// table12: %tb_4 := a 'left t_able b, 3 by 4 items, one work-item each.
__kernel void table12(__global const float* b_maxr_2, __global const long* b_b, __global float* b_tb_4) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_tb_4[k] = b_maxr_2[k / nb];
}

// k9: 12 elements, one work-item each; computes %eq_2 %f_2 %mul_4 %sub_4 %sub_5 %gt %f_3 %sub_6 %gt_2 %f_4
__kernel void k9(__global const float* b_sub_3, __global const float* b_tb_4, __global const float* b_tb_3, __global const float* b_m, __global float* b_f_3, __global float* b_f_4) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    int t_eq_2 = (b_sub_3[i] == b_tb_4[i]);
    float t_f_2 = (float)t_eq_2;
    float t_mul_4 = (b_tb_3[i] * t_f_2);
    float t_sub_4 = (b_sub_3[i] - t_mul_4);
    float t_sub_5 = (b_m[i] - t_sub_4);
    int t_gt = (t_sub_5 > 0.0f);
    float t_f_3 = (float)t_gt;
    float t_sub_6 = (b_m[i] - b_sub_3[i]);
    int t_gt_2 = (t_sub_6 > 0.0f);
    float t_f_4 = (float)t_gt_2;
    b_f_3[i] = t_f_3;
    b_f_4[i] = t_f_4;
}

// table15: %tb_5 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table15(__global const float* b_maxr_3, __global const long* b_b_3, __global float* b_tb_5) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_5[k] = b_maxr_3[k / nb];
}

// k11: 20 elements, one work-item each; computes %eq_3 %f_5 %sub_7 %mul_5
__kernel void k11(__global const float* b_s, __global const float* b_tb_5, __global float* b_f_5, __global float* b_mul_5) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_3 = (b_s[i] == b_tb_5[i]);
    float t_f_5 = (float)t_eq_3;
    float t_sub_7 = (1.0f - t_f_5);
    float t_mul_5 = (t_sub_7 * 6.0f);
    b_f_5[i] = t_f_5;
    b_mul_5[i] = t_mul_5;
}

// table17: %tb_6 := a 'right t_able b, 4 by 5 items, one work-item each.
__kernel void table17(__global const long* b_b, __global const float* b_b_4, __global float* b_tb_6) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_6[k] = b_b_4[k % nb];
}

// k13: 20 elements, one work-item each; computes %mul_6 %add_3
__kernel void k13(__global const float* b_f_5, __global const float* b_tb_6, __global const float* b_mul_5, __global float* b_add_3) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    float t_mul_6 = (b_f_5[i] * b_tb_6[i]);
    float t_add_3 = (t_mul_6 + b_mul_5[i]);
    b_add_3[i] = t_add_3;
}

// k14: 4 elements, one work-item each; computes %fl_2
__kernel void k14(__global const float* b_minr_3, __global long* b_fl_2) {
    size_t i = get_global_id(0);
    if (i >= 4) return;
    long t_fl_2 = (long)floor(b_minr_3[i]);
    b_fl_2[i] = t_fl_2;
}

// k15: 4 elements, one work-item each; computes %sub_8 %add_4
__kernel void k15(__global const float* b_maxr_3, __global const float* b_minr_4, __global float* b_add_4) {
    size_t i = get_global_id(0);
    if (i >= 4) return;
    float t_sub_8 = (b_maxr_3[i] - b_minr_4[i]);
    float t_add_4 = (1.0f + t_sub_8);
    b_add_4[i] = t_add_4;
}

// table23: %tb_7 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table23(__global const float* b_add_4, __global const long* b_b_3, __global float* b_tb_7) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_7[k] = b_add_4[k / nb];
}

// k17: 20 elements, one work-item each; computes %mul_7 %sub_9
__kernel void k17(__global const float* b_tb_7, __global const float* b_f_5, __global const float* b_s, __global float* b_sub_9) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    float t_mul_7 = (b_tb_7[i] * b_f_5[i]);
    float t_sub_9 = (b_s[i] - t_mul_7);
    b_sub_9[i] = t_sub_9;
}

// table26: %tb_8 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table26(__global const float* b_maxr_4, __global const long* b_b_3, __global float* b_tb_8) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_8[k] = b_maxr_4[k / nb];
}

// k19: 20 elements, one work-item each; computes %eq_4 %f_6 %mul_8 %sub_10
__kernel void k19(__global const float* b_sub_9, __global const float* b_tb_8, __global const float* b_tb_7, __global float* b_sub_10) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_4 = (b_sub_9[i] == b_tb_8[i]);
    float t_f_6 = (float)t_eq_4;
    float t_mul_8 = (b_tb_7[i] * t_f_6);
    float t_sub_10 = (b_sub_9[i] - t_mul_8);
    b_sub_10[i] = t_sub_10;
}

// table29: %tb_9 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table29(__global const float* b_maxr_5, __global const long* b_b_3, __global float* b_tb_9) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_9[k] = b_maxr_5[k / nb];
}

// k21: 20 elements, one work-item each; computes %eq_5 %f_7 %mul_9 %sub_11 %sub_12 %gt_3 %f_8 %mul_10
__kernel void k21(__global const float* b_sub_10, __global const float* b_tb_9, __global const float* b_tb_7, __global const float* b_s, __global float* b_f_8, __global float* b_mul_10) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_5 = (b_sub_10[i] == b_tb_9[i]);
    float t_f_7 = (float)t_eq_5;
    float t_mul_9 = (b_tb_7[i] * t_f_7);
    float t_sub_11 = (b_sub_10[i] - t_mul_9);
    float t_sub_12 = (b_s[i] - t_sub_11);
    int t_gt_3 = (t_sub_12 > 0.0f);
    float t_f_8 = (float)t_gt_3;
    float t_mul_10 = (b_s[i] * t_f_8);
    b_f_8[i] = t_f_8;
    b_mul_10[i] = t_mul_10;
}
