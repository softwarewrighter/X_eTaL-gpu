// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 8 elements, one work-item each; computes %b %c %af %x %y %z
__kernel void k0(__global const long* b_a, __global long* b_c, __global float* b_z) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    long t_b = (10L * b_a[i]);
    long t_c = (b_a[i] + t_b);
    float t_af = (float)b_a[i];
    float t_x = (t_af / 4.0f);
    float t_y = (0.5f * t_x);
    float t_z = (t_x + t_y);
    b_c[i] = t_c;
    b_z[i] = t_z;
}
