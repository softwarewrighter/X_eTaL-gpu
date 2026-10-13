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

// table1: %asp := a 'left t_able b, 3 by 4 items, one work-item each.
__kernel void table1(__global const float* b_amx, __global const long* b_o4, __global float* b_asp) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_asp[k] = b_amx[k / nb];
}

// k1: 12 elements, one work-item each; computes %aeq %ahit
__kernel void k1(__global const float* b_m, __global const float* b_asp, __global float* b_ahit) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    int t_aeq = (b_m[i] == b_asp[i]);
    float t_ahit = (float)t_aeq;
    b_ahit[i] = t_ahit;
}

// table3: %apos := a 'right t_able b, 3 by 4 items, one work-item each.
__kernel void table3(__global const long* b_o3, __global const float* b_r4, __global float* b_apos) {
    long k = get_global_id(0);
    const long nb = 4;
    if (k >= 12) return;
    b_apos[k] = b_r4[k % nb];
}

// k3: 12 elements, one work-item each; computes %aa %anh %ab %as
__kernel void k3(__global const float* b_ahit, __global const float* b_apos, __global float* b_as) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    float t_aa = (b_ahit[i] * b_apos[i]);
    float t_anh = (1.0f - b_ahit[i]);
    float t_ab = (t_anh * 5.0f);
    float t_as = (t_aa + t_ab);
    b_as[i] = t_as;
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

// k4: 3 elements, one work-item each; computes %argm
__kernel void k4(__global const float* b_amn, __global long* b_argm) {
    size_t i = get_global_id(0);
    if (i >= 3) return;
    long t_argm = (long)floor(b_amn[i]);
    b_argm[i] = t_argm;
}

// topk2_float: a mask of each line's 2 largest items, the largest taken
// out 2 times with its ties (X_eTaL-ML's sm:t_op): one work-item per
// line keeps the 2 largest distinct values it has seen, in order, then
// marks the items at or above the last of them (all of a line with
// fewer distinct values).
__kernel void topk2_float(__global const float* in, __global int* out, const long outer, const long len, const long inner) {
    long line = get_global_id(0);
    if (line >= outer * inner) return;
    long o = line / inner, i = line % inner;
    long base = o * len * inner + i;
    float best[2];
    int found = 0;
    for (long j = 0; j < len; j++) {
        float x = in[base + j * inner];
        int pos = 0, dup = 0;
        for (int b = 0; b < found; b++) {
            if (best[b] == x) { dup = 1; break; }
            if (best[b] > x) pos = b + 1;
        }
        if (dup || pos >= 2) continue;
        int last = found < 2 ? found : 2 - 1;
        for (int b = last; b > pos; b--) best[b] = best[b - 1];
        best[pos] = x;
        if (found < 2) found++;
    }
    float threshold = best[found - 1];
    for (long j = 0; j < len; j++) out[base + j * inner] = in[base + j * inner] >= threshold;
}

// k5: 12 elements, one work-item each; computes %top2
__kernel void k5(__global const int* b_k2, __global float* b_top2) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    float t_top2 = (float)b_k2[i];
    b_top2[i] = t_top2;
}

// topk1_float: a mask of each line's 1 largest items, the largest taken
// out 1 times with its ties (X_eTaL-ML's sm:t_op): one work-item per
// line keeps the 1 largest distinct values it has seen, in order, then
// marks the items at or above the last of them (all of a line with
// fewer distinct values).
__kernel void topk1_float(__global const float* in, __global int* out, const long outer, const long len, const long inner) {
    long line = get_global_id(0);
    if (line >= outer * inner) return;
    long o = line / inner, i = line % inner;
    long base = o * len * inner + i;
    float best[1];
    int found = 0;
    for (long j = 0; j < len; j++) {
        float x = in[base + j * inner];
        int pos = 0, dup = 0;
        for (int b = 0; b < found; b++) {
            if (best[b] == x) { dup = 1; break; }
            if (best[b] > x) pos = b + 1;
        }
        if (dup || pos >= 1) continue;
        int last = found < 1 ? found : 1 - 1;
        for (int b = last; b > pos; b--) best[b] = best[b - 1];
        best[pos] = x;
        if (found < 1) found++;
    }
    float threshold = best[found - 1];
    for (long j = 0; j < len; j++) out[base + j * inner] = in[base + j * inner] >= threshold;
}

// k6: 12 elements, one work-item each; computes %top1
__kernel void k6(__global const int* b_k1, __global float* b_top1) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    float t_top1 = (float)b_k1[i];
    b_top1[i] = t_top1;
}

// table12: %bsp := a 'left t_able b, 4 by 5 items, one work-item each.
__kernel void table12(__global const float* b_bmx, __global const long* b_o5, __global float* b_bsp) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_bsp[k] = b_bmx[k / nb];
}

// k8: 20 elements, one work-item each; computes %beq %bhit
__kernel void k8(__global const float* b_s, __global const float* b_bsp, __global float* b_bhit) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    int t_beq = (b_s[i] == b_bsp[i]);
    float t_bhit = (float)t_beq;
    b_bhit[i] = t_bhit;
}

// table14: %bpos := a 'right t_able b, 4 by 5 items, one work-item each.
__kernel void table14(__global const long* b_o4, __global const float* b_r5, __global float* b_bpos) {
    long k = get_global_id(0);
    const long nb = 5;
    if (k >= 20) return;
    b_bpos[k] = b_r5[k % nb];
}

// k10: 20 elements, one work-item each; computes %ba %bnh %bb %bs
__kernel void k10(__global const float* b_bhit, __global const float* b_bpos, __global float* b_bs) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    float t_ba = (b_bhit[i] * b_bpos[i]);
    float t_bnh = (1.0f - b_bhit[i]);
    float t_bb = (t_bnh * 6.0f);
    float t_bs = (t_ba + t_bb);
    b_bs[i] = t_bs;
}

// k11: 4 elements, one work-item each; computes %args
__kernel void k11(__global const float* b_bmn, __global long* b_args) {
    size_t i = get_global_id(0);
    if (i >= 4) return;
    long t_args = (long)floor(b_bmn[i]);
    b_args[i] = t_args;
}

// topk3_float: a mask of each line's 3 largest items, the largest taken
// out 3 times with its ties (X_eTaL-ML's sm:t_op): one work-item per
// line keeps the 3 largest distinct values it has seen, in order, then
// marks the items at or above the last of them (all of a line with
// fewer distinct values).
__kernel void topk3_float(__global const float* in, __global int* out, const long outer, const long len, const long inner) {
    long line = get_global_id(0);
    if (line >= outer * inner) return;
    long o = line / inner, i = line % inner;
    long base = o * len * inner + i;
    float best[3];
    int found = 0;
    for (long j = 0; j < len; j++) {
        float x = in[base + j * inner];
        int pos = 0, dup = 0;
        for (int b = 0; b < found; b++) {
            if (best[b] == x) { dup = 1; break; }
            if (best[b] > x) pos = b + 1;
        }
        if (dup || pos >= 3) continue;
        int last = found < 3 ? found : 3 - 1;
        for (int b = last; b > pos; b--) best[b] = best[b - 1];
        best[pos] = x;
        if (found < 3) found++;
    }
    float threshold = best[found - 1];
    for (long j = 0; j < len; j++) out[base + j * inner] = in[base + j * inner] >= threshold;
}

// k12: 20 elements, one work-item each; computes %top3 %kept
__kernel void k12(__global const int* b_k3, __global const float* b_s, __global float* b_top3, __global float* b_kept) {
    size_t i = get_global_id(0);
    if (i >= 20) return;
    float t_top3 = (float)b_k3[i];
    float t_kept = (b_s[i] * t_top3);
    b_top3[i] = t_top3;
    b_kept[i] = t_kept;
}
