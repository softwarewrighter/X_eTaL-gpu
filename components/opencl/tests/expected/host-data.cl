// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

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

// k0: 5 elements, one work-item each; computes %mul
__kernel void k0(__global const long* b_v, __global long* b_mul) {
    size_t i = get_global_id(0);
    if (i >= 5) return;
    long t_mul = (b_v[i] * b_v[i]);
    b_mul[i] = t_mul;
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

// transpose3: %t := o_\, the axes reversed, [3 3] to [3 3]; a gather, one work-item per result item.
__kernel void transpose3(__global const long* b_m, __global long* b_t) {
    long k = get_global_id(0);
    if (k >= 9) return;
    long r = k, src = 0;
    src += (r % 3) * 3; r /= 3;
    src += r * 1;
    b_t[k] = b_m[src];
}

// matmul_long: a by b, a's last axis (n) with b's first; one work-item per
// result item (i, j), its n products summed from the last to the first,
// as X_eTaL's '+ '* i_nner sums them.
__kernel void matmul_long(__global const long* a, __global const long* b, __global long* c, const long m, const long n, const long p) {
    long ij = get_global_id(0);
    if (ij >= m * p) return;
    long i = ij / p, j = ij % p;
    long acc = a[i * n + n - 1] * b[(n - 1) * p + j];
    for (long k = n - 2; k >= 0; k--) acc = a[i * n + k] * b[k * p + j] + acc;
    c[ij] = acc;
}

// k2: 5 elements, one work-item each; computes %f %div
__kernel void k2(__global const long* b_v) {
    size_t i = get_global_id(0);
    if (i >= 5) return;
    float t_f = (float)b_v[i];
    float t_div = (t_f / 2.0f);
}
