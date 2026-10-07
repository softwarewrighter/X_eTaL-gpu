// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 12 elements, one work-item each; computes %m1
__kernel void k0(__global const long* b_m, __global long* b_m1) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    long t_m1 = (b_m[i] * 1L);
    b_m1[i] = t_m1;
}

// reduce_axis_add_long: along one axis, one work-item per result item; its line of
// len items (stride inner) folded from the right, as X_eTaL's r_/ does.
__kernel void reduce_axis_add_long(__global const long* in, __global long* out, const long outer, const long len, const long inner) {
    long k = get_global_id(0);
    if (k >= outer * inner) return;
    long o = k / inner, i = k % inner;
    long base = o * len * inner + i;
    long acc = in[base + (len - 1) * inner];
    for (long j = len - 2; j >= 0; j--) acc = in[base + j * inner] + acc;
    out[k] = acc;
}

// reduce_axis_max_long: along one axis, one work-item per result item; its line of
// len items (stride inner) folded from the right, as X_eTaL's r_/ does.
__kernel void reduce_axis_max_long(__global const long* in, __global long* out, const long outer, const long len, const long inner) {
    long k = get_global_id(0);
    if (k >= outer * inner) return;
    long o = k / inner, i = k % inner;
    long base = o * len * inner + i;
    long acc = in[base + (len - 1) * inner];
    for (long j = len - 2; j >= 0; j--) acc = max(in[base + j * inner], acc);
    out[k] = acc;
}

// k1: 12 elements, one work-item each; computes %mf %f
__kernel void k1(__global const long* b_m, __global float* b_f) {
    size_t i = get_global_id(0);
    if (i >= 12) return;
    float t_mf = (float)b_m[i];
    float t_f = (t_mf / 8.0f);
    b_f[i] = t_f;
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

// reduce_add_float: a tree reduction in local memory, one partial per work-group;
// launched again on the partials until one value is left.
__kernel void reduce_add_float(__global const float* in, __global float* out, const long n, __local float* s) {
    size_t gid = get_global_id(0), lid = get_local_id(0), ls = get_local_size(0);
    s[lid] = gid < (size_t)n ? in[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t h = ls / 2; h > 0; h >>= 1) {
        if (lid < h) s[lid] = s[lid] + s[lid + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out[get_group_id(0)] = s[0];
}
