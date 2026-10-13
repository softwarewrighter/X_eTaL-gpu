// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 8 elements, one work-item each; computes %div %mul %mul_2 %add
__kernel void k0(__global const float* b_l, __global const float* b_r_2, __global float* b_add) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_div = (b_l[i] / 4.0f);
    float t_mul = (1.0f * b_r_2[i]);
    float t_mul_2 = (2.0f * t_div);
    float t_add = (t_mul_2 + t_mul);
    b_add[i] = t_add;
}
