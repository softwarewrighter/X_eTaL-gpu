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
__kernel void k5(__global const float* b_maxr_2, __global const float* b_minr_2, __global float* b_add_2) {
    size_t i = get_global_id(0);
    if (i >= 3) return;
    float t_sub_2 = (b_maxr_2[i] - b_minr_2[i]);
    float t_add_2 = (1.0f + t_sub_2);
    b_add_2[i] = t_add_2;
}

// table10: %tb_3 := a 'left t_able b, 3 by 4 items, one work-item each.
__kernel void table10(__global const float* b_add_2, __global const long* b_b_3, __global float* b_tb_3) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_tb_3[k] = b_add_2[k / nb];
}

// table12: %tb_4 := a 'left t_able b, 3 by 4 items, one work-item each.
__kernel void table12(__global const float* b_maxr_3, __global const long* b_b_4, __global float* b_tb_4) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_tb_4[k] = b_maxr_3[k / nb];
}

// k8: 12 elements, one work-item each; computes %eq_2 %f_2 %mul_3 %sub_3
__kernel void k8(__global const float* b_m, __global const float* b_tb_4, __global const float* b_tb_3, __global float* b_sub_3) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    int t_eq_2 = (b_m[i] == b_tb_4[i]);
    float t_f_2 = (float)t_eq_2;
    float t_mul_3 = (b_tb_3[i] * t_f_2);
    float t_sub_3 = (b_m[i] - t_mul_3);
    b_sub_3[i] = t_sub_3;
}

// table15: %tb_5 := a 'left t_able b, 3 by 4 items, one work-item each.
__kernel void table15(__global const float* b_maxr_4, __global const long* b_b_5, __global float* b_tb_5) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_tb_5[k] = b_maxr_4[k / nb];
}

// k10: 12 elements, one work-item each; computes %eq_3 %f_3 %mul_4 %sub_4 %sub_5 %gt %f_4
__kernel void k10(__global const float* b_sub_3, __global const float* b_tb_5, __global const float* b_tb_3, __global const float* b_m, __global float* b_f_4) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    int t_eq_3 = (b_sub_3[i] == b_tb_5[i]);
    float t_f_3 = (float)t_eq_3;
    float t_mul_4 = (b_tb_3[i] * t_f_3);
    float t_sub_4 = (b_sub_3[i] - t_mul_4);
    float t_sub_5 = (b_m[i] - t_sub_4);
    int t_gt = (t_sub_5 > 0.0f);
    float t_f_4 = (float)t_gt;
    b_f_4[i] = t_f_4;
}

// k11: 3 elements, one work-item each; computes %sub_6 %add_3
__kernel void k11(__global const float* b_maxr_5, __global const float* b_minr_3, __global float* b_add_3) {
    size_t i = get_global_id(0);
    if (i >= 3) return;
    float t_sub_6 = (b_maxr_5[i] - b_minr_3[i]);
    float t_add_3 = (1.0f + t_sub_6);
    b_add_3[i] = t_add_3;
}

// table20: %tb_6 := a 'left t_able b, 3 by 4 items, one work-item each.
__kernel void table20(__global const float* b_add_3, __global const long* b_b_6, __global float* b_tb_6) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_tb_6[k] = b_add_3[k / nb];
}

// table22: %tb_7 := a 'left t_able b, 3 by 4 items, one work-item each.
__kernel void table22(__global const float* b_maxr_6, __global const long* b_b_7, __global float* b_tb_7) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_tb_7[k] = b_maxr_6[k / nb];
}

// k14: 12 elements, one work-item each; computes %eq_4 %f_5 %mul_5 %sub_7 %sub_8 %gt_2 %f_6
__kernel void k14(__global const float* b_m, __global const float* b_tb_7, __global const float* b_tb_6, __global float* b_f_6) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    int t_eq_4 = (b_m[i] == b_tb_7[i]);
    float t_f_5 = (float)t_eq_4;
    float t_mul_5 = (b_tb_6[i] * t_f_5);
    float t_sub_7 = (b_m[i] - t_mul_5);
    float t_sub_8 = (b_m[i] - t_sub_7);
    int t_gt_2 = (t_sub_8 > 0.0f);
    float t_f_6 = (float)t_gt_2;
    b_f_6[i] = t_f_6;
}

// table25: %tb_8 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table25(__global const float* b_maxr_7, __global const long* b_b_8, __global float* b_tb_8) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_8[k] = b_maxr_7[k / nb];
}

// k16: 20 elements, one work-item each; computes %eq_5 %f_7 %sub_9 %mul_6
__kernel void k16(__global const float* b_s, __global const float* b_tb_8, __global float* b_f_7, __global float* b_mul_6) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_5 = (b_s[i] == b_tb_8[i]);
    float t_f_7 = (float)t_eq_5;
    float t_sub_9 = (1.0f - t_f_7);
    float t_mul_6 = (t_sub_9 * 6.0f);
    b_f_7[i] = t_f_7;
    b_mul_6[i] = t_mul_6;
}

// table27: %tb_9 := a 'right t_able b, 4 by 5 items, one work-item each.
__kernel void table27(__global const long* b_a_2, __global const float* b_b_9, __global float* b_tb_9) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_9[k] = b_b_9[k % nb];
}

// k18: 20 elements, one work-item each; computes %mul_7 %add_4
__kernel void k18(__global const float* b_f_7, __global const float* b_tb_9, __global const float* b_mul_6, __global float* b_add_4) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    float t_mul_7 = (b_f_7[i] * b_tb_9[i]);
    float t_add_4 = (t_mul_7 + b_mul_6[i]);
    b_add_4[i] = t_add_4;
}

// k19: 4 elements, one work-item each; computes %fl_2
__kernel void k19(__global const float* b_minr_4, __global long* b_fl_2) {
    size_t i = get_global_id(0);
    if (i >= 4) return;
    long t_fl_2 = (long)floor(b_minr_4[i]);
    b_fl_2[i] = t_fl_2;
}

// k20: 4 elements, one work-item each; computes %sub_10 %add_5
__kernel void k20(__global const float* b_maxr_8, __global const float* b_minr_5, __global float* b_add_5) {
    size_t i = get_global_id(0);
    if (i >= 4) return;
    float t_sub_10 = (b_maxr_8[i] - b_minr_5[i]);
    float t_add_5 = (1.0f + t_sub_10);
    b_add_5[i] = t_add_5;
}

// table34: %tb_10 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table34(__global const float* b_add_5, __global const long* b_b_10, __global float* b_tb_10) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_10[k] = b_add_5[k / nb];
}

// table36: %tb_11 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table36(__global const float* b_maxr_9, __global const long* b_b_11, __global float* b_tb_11) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_11[k] = b_maxr_9[k / nb];
}

// k23: 20 elements, one work-item each; computes %eq_6 %f_8 %mul_8 %sub_11
__kernel void k23(__global const float* b_s, __global const float* b_tb_11, __global const float* b_tb_10, __global float* b_sub_11) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_6 = (b_s[i] == b_tb_11[i]);
    float t_f_8 = (float)t_eq_6;
    float t_mul_8 = (b_tb_10[i] * t_f_8);
    float t_sub_11 = (b_s[i] - t_mul_8);
    b_sub_11[i] = t_sub_11;
}

// table39: %tb_12 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table39(__global const float* b_maxr_10, __global const long* b_b_12, __global float* b_tb_12) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_12[k] = b_maxr_10[k / nb];
}

// k25: 20 elements, one work-item each; computes %eq_7 %f_9 %mul_9 %sub_12
__kernel void k25(__global const float* b_sub_11, __global const float* b_tb_12, __global const float* b_tb_10, __global float* b_sub_12) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_7 = (b_sub_11[i] == b_tb_12[i]);
    float t_f_9 = (float)t_eq_7;
    float t_mul_9 = (b_tb_10[i] * t_f_9);
    float t_sub_12 = (b_sub_11[i] - t_mul_9);
    b_sub_12[i] = t_sub_12;
}

// table42: %tb_13 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table42(__global const float* b_maxr_11, __global const long* b_b_13, __global float* b_tb_13) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_13[k] = b_maxr_11[k / nb];
}

// k27: 20 elements, one work-item each; computes %eq_8 %f_10 %mul_10 %sub_13 %sub_14 %gt_3 %f_11
__kernel void k27(__global const float* b_sub_12, __global const float* b_tb_13, __global const float* b_tb_10, __global const float* b_s, __global float* b_f_11) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_8 = (b_sub_12[i] == b_tb_13[i]);
    float t_f_10 = (float)t_eq_8;
    float t_mul_10 = (b_tb_10[i] * t_f_10);
    float t_sub_13 = (b_sub_12[i] - t_mul_10);
    float t_sub_14 = (b_s[i] - t_sub_13);
    int t_gt_3 = (t_sub_14 > 0.0f);
    float t_f_11 = (float)t_gt_3;
    b_f_11[i] = t_f_11;
}

// k28: 4 elements, one work-item each; computes %sub_15 %add_6
__kernel void k28(__global const float* b_maxr_12, __global const float* b_minr_6, __global float* b_add_6) {
    size_t i = get_global_id(0);
    if (i >= 4) return;
    float t_sub_15 = (b_maxr_12[i] - b_minr_6[i]);
    float t_add_6 = (1.0f + t_sub_15);
    b_add_6[i] = t_add_6;
}

// table47: %tb_14 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table47(__global const float* b_add_6, __global const long* b_b_14, __global float* b_tb_14) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_14[k] = b_add_6[k / nb];
}

// table49: %tb_15 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table49(__global const float* b_maxr_13, __global const long* b_b_15, __global float* b_tb_15) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_15[k] = b_maxr_13[k / nb];
}

// k31: 20 elements, one work-item each; computes %eq_9 %f_12 %mul_11 %sub_16
__kernel void k31(__global const float* b_s, __global const float* b_tb_15, __global const float* b_tb_14, __global float* b_sub_16) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_9 = (b_s[i] == b_tb_15[i]);
    float t_f_12 = (float)t_eq_9;
    float t_mul_11 = (b_tb_14[i] * t_f_12);
    float t_sub_16 = (b_s[i] - t_mul_11);
    b_sub_16[i] = t_sub_16;
}

// table52: %tb_16 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table52(__global const float* b_maxr_14, __global const long* b_b_16, __global float* b_tb_16) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_16[k] = b_maxr_14[k / nb];
}

// k33: 20 elements, one work-item each; computes %eq_10 %f_13 %mul_12 %sub_17
__kernel void k33(__global const float* b_sub_16, __global const float* b_tb_16, __global const float* b_tb_14, __global float* b_sub_17) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_10 = (b_sub_16[i] == b_tb_16[i]);
    float t_f_13 = (float)t_eq_10;
    float t_mul_12 = (b_tb_14[i] * t_f_13);
    float t_sub_17 = (b_sub_16[i] - t_mul_12);
    b_sub_17[i] = t_sub_17;
}

// table55: %tb_17 := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table55(__global const float* b_maxr_15, __global const long* b_b_17, __global float* b_tb_17) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_tb_17[k] = b_maxr_15[k / nb];
}

// k35: 20 elements, one work-item each; computes %eq_11 %f_14 %mul_13 %sub_18 %sub_19 %gt_4 %f_15 %mul_14
__kernel void k35(__global const float* b_sub_17, __global const float* b_tb_17, __global const float* b_tb_14, __global const float* b_s, __global float* b_mul_14) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_eq_11 = (b_sub_17[i] == b_tb_17[i]);
    float t_f_14 = (float)t_eq_11;
    float t_mul_13 = (b_tb_14[i] * t_f_14);
    float t_sub_18 = (b_sub_17[i] - t_mul_13);
    float t_sub_19 = (b_s[i] - t_sub_18);
    int t_gt_4 = (t_sub_19 > 0.0f);
    float t_f_15 = (float)t_gt_4;
    float t_mul_14 = (b_s[i] * t_f_15);
    b_mul_14[i] = t_mul_14;
}
