// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// rotate0: %rot := 1 o_-_1 (1 copy, 1 line of 5 along the axis, 1 apart); a gather, one work-item per result item.
__constant long rotate0_amounts[1] = { 1L };
__kernel void rotate0(__global const long* b_v, __global long* b_rot) {
    long k = get_global_id(0);
    if (k >= 5) return;
    long c = k / 5, r = k % 5;
    long o = r / (5 * 1), q = r % (5 * 1);
    long j = q / 1, i = q % 1;
    long s = xetal_imod_long(j + rotate0_amounts[c], 5);
    b_rot[k] = b_v[(o * 5 + s) * 1 + i];
}

// rotate1: %rot_2 := -1 0 1 o_-_1 (3 copies, 1 line of 5 along the axis, 1 apart); a gather, one work-item per result item.
__constant long rotate1_amounts[3] = { -1L, 0L, 1L };
__kernel void rotate1(__global const long* b_v, __global long* b_rot_2) {
    long k = get_global_id(0);
    if (k >= 15) return;
    long c = k / 5, r = k % 5;
    long o = r / (5 * 1), q = r % (5 * 1);
    long j = q / 1, i = q % 1;
    long s = xetal_imod_long(j + rotate1_amounts[c], 5);
    b_rot_2[k] = b_v[(o * 5 + s) * 1 + i];
}

// rotate2: %rot_3 := -1 0 1 o_-_3 (3 copies, 6 lines of 4 along the axis, 1 apart); a gather, one work-item per result item.
__constant long rotate2_amounts[3] = { -1L, 0L, 1L };
__kernel void rotate2(__global const float* b_x, __global float* b_rot_3) {
    long k = get_global_id(0);
    if (k >= 72) return;
    long c = k / 24, r = k % 24;
    long o = r / (4 * 1), q = r % (4 * 1);
    long j = q / 1, i = q % 1;
    long s = xetal_imod_long(j + rotate2_amounts[c], 4);
    b_rot_3[k] = b_x[(o * 4 + s) * 1 + i];
}

// rotate3: %rot_4 := -1 0 1 o_-_3 (3 copies, 6 lines of 3 along the axis, 4 apart); a gather, one work-item per result item.
__constant long rotate3_amounts[3] = { -1L, 0L, 1L };
__kernel void rotate3(__global const float* b_rot_3, __global float* b_rot_4) {
    long k = get_global_id(0);
    if (k >= 216) return;
    long c = k / 72, r = k % 72;
    long o = r / (3 * 4), q = r % (3 * 4);
    long j = q / 4, i = q % 4;
    long s = xetal_imod_long(j + rotate3_amounts[c], 3);
    b_rot_4[k] = b_rot_3[(o * 3 + s) * 4 + i];
}

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

// rotate8: %rot_5 := -1 0 1 o_-_3 (3 copies, 6 lines of 4 along the axis, 1 apart); a gather, one work-item per result item.
__constant long rotate8_amounts[3] = { -1L, 0L, 1L };
__kernel void rotate8(__global const float* b_x, __global float* b_rot_5) {
    long k = get_global_id(0);
    if (k >= 72) return;
    long c = k / 24, r = k % 24;
    long o = r / (4 * 1), q = r % (4 * 1);
    long j = q / 1, i = q % 1;
    long s = xetal_imod_long(j + rotate8_amounts[c], 4);
    b_rot_5[k] = b_x[(o * 4 + s) * 1 + i];
}

// rotate9: %rot_6 := -1 0 1 o_-_3 (3 copies, 6 lines of 3 along the axis, 4 apart); a gather, one work-item per result item.
__constant long rotate9_amounts[3] = { -1L, 0L, 1L };
__kernel void rotate9(__global const float* b_rot_5, __global float* b_rot_6) {
    long k = get_global_id(0);
    if (k >= 216) return;
    long c = k / 72, r = k % 72;
    long o = r / (3 * 4), q = r % (3 * 4);
    long j = q / 4, i = q % 4;
    long s = xetal_imod_long(j + rotate9_amounts[c], 3);
    b_rot_6[k] = b_rot_5[(o * 3 + s) * 4 + i];
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

// transpose15: %tr := 2 1 3 t_ranspose, [2 2 2] to [2 2 2]; a gather, one work-item per result item.
__kernel void transpose15(__global const float* b_mm, __global float* b_tr) {
    long k = get_global_id(0);
    if (k >= 8) return;
    long r = k, src = 0;
    src += (r % 2) * 1; r /= 2;
    src += (r % 2) * 4; r /= 2;
    src += r * 2;
    b_tr[k] = b_mm[src];
}
