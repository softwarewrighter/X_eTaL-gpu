// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 8 elements, one work-item each; computes %div %sub %gt %f %mul %max %sub_2 %max_2 %mod %mul_2
__kernel void k0(__global const float* b_l, __global const long* b_l_2, __global float* b_sub, __global float* b_mul, __global float* b_max, __global long* b_max_2, __global long* b_mul_2) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_div = (b_l[i] / 4.0f);
    float t_sub = (t_div - 1.0f);
    int t_gt = (t_sub > 0.5f);
    float t_f = (float)t_gt;
    float t_mul = (t_sub * t_f);
    float t_max = fmax(t_sub, 0.0f);
    long t_sub_2 = (b_l_2[i] - 4L);
    long t_max_2 = max(t_sub_2, 0L);
    long t_mod = xetal_imod_long(t_sub_2, 2L);
    long t_mul_2 = (t_sub_2 * t_mod);
    b_sub[i] = t_sub;
    b_mul[i] = t_mul;
    b_max[i] = t_max;
    b_max_2[i] = t_max_2;
    b_mul_2[i] = t_mul_2;
}
