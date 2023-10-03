
#define MAXSIZE 1024
__kernel void antirectifier_kernel(const __global half* input0, __global half* output) {
	half mean[MAXSIZE] = {0};
	// Compute the mean NHWC
	int W = get_global_size(0);
	int H = get_global_size(1);
	int C = get_global_size(2);
	for(int cc = 0; cc < C; cc++) {
		for(int ww = 0; ww < W; ww++) {
			for(int hh = 0; hh < H; hh++) {
				int x = ww+hh*W+cc*H*W;
				mean[cc] += input0[x];
			}
		}
		mean[cc] /= (half)W*(half)H;
	}
	// Compute the absolute value of pos and neg parts
	for(int cc = 0; cc < C; cc++) {
		for(int ww = 0; ww < W; ww++) {
			for(int hh = 0; hh < H; hh++) {
				int x = ww+hh*W+cc*H*W;
				int yp = x;
				int ym = x + C*W*H;
				output[yp] = (input0[x]-mean[cc]) > 0.0 ? input0[x]-mean[cc] : 0.0;
				output[ym] = (input0[x]-mean[cc]) < 0.0 ? mean[cc]-input0[x] : 0.0;
			}
		}
	}
}