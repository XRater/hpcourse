__kernel void convolution(__global float * a, __global float * b, __global float * c, int n, int m)
{
    int i = get_global_id(0);
    int j = get_global_id(1);    
   
    if (i >= n || j >= n) {
        return;
    }
    float res = 0;
    int hm = (m - 1) / 2;
    for (int k = -hm; k <= hm; k++) {
        for (int l = -hm; l <= hm; l++) {
            if (i + k < 0 || i + k >= n || j + l < 0 || j + l >= n) {
                continue;
            }
            res += a[(i + k) * n + (j + l)] * b[(k + hm) * m + (l + hm)];
        }
    }
    c[i * n + j] = res;
}
