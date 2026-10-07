// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 8 elements, one work-item each; computes %xw %s %pos
__kernel void k0(__global const long* b_x, __global const long* b_w, __global long* b_pos) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    long t_xw = (b_x[i] * b_w[i]);
    long t_s = (b_x[i] + t_xw);
    long t_pos = max(t_s, 0L);
    b_pos[i] = t_pos;
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

// k1: 8 elements, one work-item each; computes %f %fg %fs %fpos
__kernel void k1(__global const float* b_r, __global const float* b_g, __global float* b_fpos) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_f = (b_r[i] / 4.0f);
    float t_fg = (t_f * b_g[i]);
    float t_fs = (t_f + t_fg);
    float t_fpos = fmax(t_fs, 0.0f);
    b_fpos[i] = t_fpos;
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
