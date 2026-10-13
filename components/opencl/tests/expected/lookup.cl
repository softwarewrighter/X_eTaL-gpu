// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// lookup0: %sel := positions s_elect_1 a (1 position, 1 line of 5 along the axis, 1 apart); a gather, one work-item per result item.
__kernel void lookup0(__global const long* b_v, __global long* b_sel) {
    long k = get_global_id(0);
    if (k >= 1) return;
    long o = k / (1 * 1), r = k % (1 * 1);
    long c = r / 1, i = r % 1;
    long p = (long)2L;
    p = p < 1 ? 1 : (p > 5 ? 5 : p);
    b_sel[k] = b_v[(o * 5 + p - 1) * 1 + i];
}

// lookup1: %sel_2 := positions s_elect_1 a (3 positions, 1 line of 5 along the axis, 1 apart); a gather, one work-item per result item.
__kernel void lookup1(__global const long* b_ids_2, __global const long* b_v, __global long* b_sel_2) {
    long k = get_global_id(0);
    if (k >= 3) return;
    long o = k / (3 * 1), r = k % (3 * 1);
    long c = r / 1, i = r % 1;
    long p = (long)b_ids_2[c];
    p = p < 1 ? 1 : (p > 5 ? 5 : p);
    b_sel_2[k] = b_v[(o * 5 + p - 1) * 1 + i];
}

// lookup2: %sel_3 := positions s_elect_1 a (4 positions, 1 line of 5 along the axis, 1 apart); a gather, one work-item per result item.
__kernel void lookup2(__global const long* b_ids_3, __global const long* b_v, __global long* b_sel_3) {
    long k = get_global_id(0);
    if (k >= 4) return;
    long o = k / (4 * 1), r = k % (4 * 1);
    long c = r / 1, i = r % 1;
    long p = (long)b_ids_3[c];
    p = p < 1 ? 1 : (p > 5 ? 5 : p);
    b_sel_3[k] = b_v[(o * 5 + p - 1) * 1 + i];
}

// lookup3: %sel_4 := positions s_elect_1 a (2 positions, 1 line of 3 along the axis, 2 apart); a gather, one work-item per result item.
__kernel void lookup3(__global const long* b_ids_4, __global const long* b_m, __global long* b_sel_4) {
    long k = get_global_id(0);
    if (k >= 4) return;
    long o = k / (2 * 2), r = k % (2 * 2);
    long c = r / 2, i = r % 2;
    long p = (long)b_ids_4[c];
    p = p < 1 ? 1 : (p > 3 ? 3 : p);
    b_sel_4[k] = b_m[(o * 3 + p - 1) * 2 + i];
}

// lookup4: %sel_5 := positions s_elect_2 a (2 positions, 3 lines of 2 along the axis, 1 apart); a gather, one work-item per result item.
__kernel void lookup4(__global const long* b_ids_5, __global const long* b_m, __global long* b_sel_5) {
    long k = get_global_id(0);
    if (k >= 6) return;
    long o = k / (2 * 1), r = k % (2 * 1);
    long c = r / 1, i = r % 1;
    long p = (long)b_ids_5[c];
    p = p < 1 ? 1 : (p > 2 ? 2 : p);
    b_sel_5[k] = b_m[(o * 2 + p - 1) * 1 + i];
}

// lookup5: %sel_6 := positions s_elect_1 a (6 positions, 1 line of 5 along the axis, 4 apart); a gather, one work-item per result item.
__kernel void lookup5(__global const long* b_ids_6, __global const float* b_t, __global float* b_sel_6) {
    long k = get_global_id(0);
    if (k >= 24) return;
    long o = k / (6 * 4), r = k % (6 * 4);
    long c = r / 4, i = r % 4;
    long p = (long)b_ids_6[c];
    p = p < 1 ? 1 : (p > 5 ? 5 : p);
    b_sel_6[k] = b_t[(o * 5 + p - 1) * 4 + i];
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
