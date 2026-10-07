// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 8 elements, one work-item each; computes %b %c
__kernel void k0(__global const long* b_a, __global long* b_b, __global long* b_c) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    long t_b = (2L * b_a[i]);
    long t_c = (b_a[i] + t_b);
    b_b[i] = t_b;
    b_c[i] = t_c;
}

// reduce_add_long: a tree reduction in local memory, one partial per work-group;
// launched again on the partials until one value is left.
__kernel void reduce_add_long(__global const long* in, __global long* out, const long n, __local long* s) {
    size_t gid = get_global_id(0), lid = get_local_id(0), ls = get_local_size(0);
    s[lid] = gid < (size_t)n ? in[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t h = ls / 2; h > 0; h >>= 1) {
        if (lid < h) s[lid] = s[lid] + s[lid + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out[get_group_id(0)] = s[0];
}

// reduce_max_long: a tree reduction in local memory, one partial per work-group;
// launched again on the partials until one value is left.
__kernel void reduce_max_long(__global const long* in, __global long* out, const long n, __local long* s) {
    size_t gid = get_global_id(0), lid = get_local_id(0), ls = get_local_size(0);
    s[lid] = gid < (size_t)n ? in[gid] : LONG_MIN;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t h = ls / 2; h > 0; h >>= 1) {
        if (lid < h) s[lid] = max(s[lid], s[lid + h]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out[get_group_id(0)] = s[0];
}

// k1: 8 elements, one work-item each; computes %ab
__kernel void k1(__global const long* b_a, __global const long* b_b, __global long* b_ab) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    long t_ab = (b_a[i] * b_b[i]);
    b_ab[i] = t_ab;
}

// k2: 8 elements, one work-item each; computes %af %f
__kernel void k2(__global const long* b_a, __global float* b_f) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_af = (float)b_a[i];
    float t_f = (t_af / 8.0f);
    b_f[i] = t_f;
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

// k3: 8 elements, one work-item each; computes %ff
__kernel void k3(__global const float* b_f, __global float* b_ff) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_ff = (b_f[i] * b_f[i]);
    b_ff[i] = t_ff;
}
