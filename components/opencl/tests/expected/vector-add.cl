// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 8 elements, one work-item each; computes %mul %add %div %mul_2 %add_2
__kernel void k0(__global const long* b_r, __global const long* b_a, __global const float* b_l_2, __global long* b_add, __global float* b_add_2) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    long t_mul = (10L * b_r[i]);
    long t_add = (b_a[i] + t_mul);
    float t_div = (b_l_2[i] / 4.0f);
    float t_mul_2 = (0.5f * t_div);
    float t_add_2 = (t_div + t_mul_2);
    b_add[i] = t_add;
    b_add_2[i] = t_add_2;
}
