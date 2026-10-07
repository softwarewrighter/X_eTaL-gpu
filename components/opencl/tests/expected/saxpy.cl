// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 8 elements, one work-item each; computes %x %y %ax %s
__kernel void k0(__global const float* b_r, __global const float* b_y0, __global float* b_s) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_x = (b_r[i] / 4.0f);
    float t_y = (1.0f * b_y0[i]);
    float t_ax = (2.0f * t_x);
    float t_s = (t_ax + t_y);
    b_s[i] = t_s;
}
