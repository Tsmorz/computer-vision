#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#define CHECK_LAUNCH assert(cudaDeviceSynchronize() == cudaSuccess)
#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W
#define CHANNEL_STRIDE 32


template <typename scalar_t>
__device__
scalar_t get_1d_padding(torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>t,
                        int i, int j, bool mirror) {
    if (i < 0 && mirror)
        i = -i;
    if (i < 0 && !mirror)
        return 0;
    if (j < 0 && mirror)
        j = -j;
    if (j < 0 && !mirror)
        return 0;
    return t[i][j];
}

/**
 * A naive CUDA implementation of NCC that does not use shared memory, but is significantly faster than the serial version
 * @tparam scalar_t
 * @param f
 * @param g
 */
template <typename scalar_t>
__global__ void k_norm_cross_corr(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> in,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> f,
    bool mirror,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int f_h = f.size(0);
    const int f_h_r = f_h / 2;
    const int f_w = f.size(1);
    const int f_w_r = f_w / 2;
    scalar_t in_norm = 0.;
    scalar_t f_norm = 0.;
    scalar_t result = 0.;

    for (int i = 0; i < f_h; i++) {
    for (int j = 0; j < f_w; j++) {
        scalar_t curr_in = get_1d_padding(in, x + i - f_h_r, y + j - f_w_r, mirror);
        scalar_t curr_f = f[i][j];
        result += curr_in * curr_f;
        in_norm += curr_in * curr_in;
        f_norm += curr_f * curr_f;
    }}
    in_norm = sqrt(in_norm);
    f_norm = sqrt(f_norm);
    result /= (in_norm * f_norm);

    out[x][y] = result;
}


torch::Tensor NCC(const torch::Tensor& in, const torch::Tensor& f, bool mirror) {
    const int h = in.size(0);
    const int w = in.size(1);
    const dim3 blocks((h + BLOCK_H - 1) / BLOCK_H, (w + BLOCK_W - 1) / BLOCK_W);
    const dim3 threads(BLOCK_H, BLOCK_W);
    auto out = torch::zeros_like(in);
    AT_DISPATCH_FLOATING_TYPES(in.type(), "k_norm_cross_corr", ([&] {
        k_norm_cross_corr<scalar_t><<<blocks, threads>>>(
                in.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                f.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                mirror,
                out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
    CHECK_LAUNCH;
    return out;
}

template<typename scalar_t>
__global__
void k_corner_nms(const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> in,
                  const int window_rad,
                  torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_size = in.size(0);
    const int y_size = in.size(1);
    if (x < x_size && y < y_size) {
        scalar_t max = 0.;
        scalar_t self_val = in[x][y];
        for (int i = - window_rad; i <= window_rad; i++) {
        for (int j = - window_rad; j <= window_rad; j++) {
            if (x_size >i + x >= 0 && y_size > j + y >= 0) {
                scalar_t curr_val = in[x + i][y + j];
                if (max < curr_val)
                    max = curr_val;
        }}}
        out[x][y] = max == self_val ? self_val : 0.;
    }
}


torch::Tensor corner_NMS(const torch::Tensor& in, const int window_rad) {
    const int h = in.size(0);
    const int w = in.size(1);
    const dim3 blocks((h + BLOCK_H - 1) / BLOCK_H, (w + BLOCK_W - 1) / BLOCK_W);
    const dim3 threads(BLOCK_H, BLOCK_W);
    auto out = torch::zeros_like(in);
    AT_DISPATCH_FLOATING_TYPES(in.type(), "k_corner_nms", ([&] {
        k_corner_nms<scalar_t><<<blocks, threads>>>(
                in.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                window_rad,
                out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
    CHECK_LAUNCH;
    return out;
}

template<typename scalar_t>
__global__
void k_harris_corner(const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> I_x,
                     const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> I_y,
                     const int radius, const float k,
                     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x_size = I_x.size(0);
        const int y_size = I_y.size(1);
        if (x < x_size && y < y_size) {
            scalar_t sum_I_xx = 0.;
            scalar_t sum_I_yy = 0.;
            scalar_t sum_I_xy = 0.;
            for (int i = - radius; i <= radius; i++) {
            for (int j = - radius; j <= radius; j++) {
            if (i + x >= 0 && j + y >= 0) {
                scalar_t curr_I_x = I_x[i + x][j + y];
                scalar_t curr_I_y = I_y[i + x][j + y];
                sum_I_xx += curr_I_x * curr_I_x;
                sum_I_yy += curr_I_y * curr_I_y;
                sum_I_xy += curr_I_x * curr_I_y;
            }}}
            scalar_t det = sum_I_xx * sum_I_yy - sum_I_xy * sum_I_xy;
            scalar_t trace = sum_I_xx + sum_I_yy;
            out[x][y] =  det - k * trace * trace;
        }
}



torch::Tensor harris_corner_detector(const torch::Tensor& I_x,
                                     const torch::Tensor& I_y,
                                     const int width, const float k) {

    const int h = I_x.size(0);
    const int w = I_y.size(1);
    const dim3 blocks((h + BLOCK_H - 1) / BLOCK_H, (w + BLOCK_W - 1) / BLOCK_W);
    const dim3 threads(BLOCK_H, BLOCK_W);
    auto out = torch::zeros_like(I_x);
    AT_DISPATCH_FLOATING_TYPES(I_x.type(), "k_harris_corner", ([&] {
        k_harris_corner<scalar_t><<<blocks, threads>>>(
                I_x.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                I_y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                width, k,
                out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
    CHECK_LAUNCH;
    return out;
}