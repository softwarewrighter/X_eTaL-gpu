// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }

// k0: 8 elements, one work-item each; computes %q %x %m %mf %th %relu %n %relun %mask %kept
__kernel void k0(__global const float* b_r, __global const long* b_n0, __global float* b_x, __global float* b_th, __global float* b_relu, __global long* b_relun, __global long* b_kept) {
    size_t i = get_global_id(0);
    if (i >= 8) return;
    float t_q = (b_r[i] / 4.0f);
    float t_x = (t_q - 1.0f);
    int t_m = (t_x > 0.5f);
    float t_mf = (float)t_m;
    float t_th = (t_x * t_mf);
    float t_relu = fmax(t_x, 0.0f);
    long t_n = (b_n0[i] - 4L);
    long t_relun = max(t_n, 0L);
    long t_mask = xetal_imod_long(t_n, 2L);
    long t_kept = (t_n * t_mask);
    b_x[i] = t_x;
    b_th[i] = t_th;
    b_relu[i] = t_relu;
    b_relun[i] = t_relun;
    b_kept[i] = t_kept;
}
