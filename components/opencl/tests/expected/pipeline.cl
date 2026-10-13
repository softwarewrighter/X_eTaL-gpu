// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 8 elements, one work-item each; computes %mul %add %max
__kernel void k0(__global const long* b_x, __global const long* b_w, __global long* b_max) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    long t_mul = (b_x[i] * b_w[i]);
    long t_add = (b_x[i] + t_mul);
    long t_max = max(t_add, 0L);
    b_max[i] = t_max;
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

// k1: 8 elements, one work-item each; computes %div %mul_2 %add_2 %max_2
__kernel void k1(__global const float* b_l, __global const float* b_g, __global float* b_max_2) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_div = (b_l[i] / 4.0f);
    float t_mul_2 = (t_div * b_g[i]);
    float t_add_2 = (t_div + t_mul_2);
    float t_max_2 = fmax(t_add_2, 0.0f);
    b_max_2[i] = t_max_2;
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
