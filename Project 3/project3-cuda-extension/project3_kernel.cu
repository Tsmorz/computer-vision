#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>
#define CHECK_LAUNCH assert(cudaDeviceSynchronize() == cudaSuccess)
#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W


template<typename scalar_t>
__device__
scalar_t local_patch_ncc(const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>& left_img,
                      const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>& right_img,
                      int left_x,
                      int left_y,
                      int right_x,
                      int right_y,
                      int patch_rad) {
    int r_left = left_img.size(0);
    int c_left = left_img.size(1);
    int r_right = right_img.size(0);
    int c_right = right_img.size(1);
    int left_patch_x_begin = left_x - patch_rad < 0 ? left_x : patch_rad;
    int left_patch_x_end = left_x + patch_rad > r_left ? r_left - left_x : patch_rad;
    int left_patch_y_begin = left_y - patch_rad < 0 ? left_y : patch_rad;
    int left_patch_y_end = left_y + patch_rad > c_left ? c_left - left_y : patch_rad;
    int right_patch_x_begin = right_x - patch_rad < 0 ? right_x : patch_rad;
    int right_patch_x_end = right_x + patch_rad > r_right ? r_right - right_x : patch_rad;
    int right_patch_y_begin = right_y - patch_rad < 0 ? right_y : patch_rad;
    int right_patch_y_end = right_y + patch_rad > c_right ? c_right - right_y : patch_rad;

    int patch_x_begin = min(left_patch_x_begin, right_patch_x_begin);
    int patch_x_end = min(left_patch_x_end, right_patch_x_end);
    int patch_y_begin = min(left_patch_y_begin, right_patch_y_begin);
    int patch_y_end = min(left_patch_y_end, right_patch_y_end);
    scalar_t left_norm = 0.;
    scalar_t right_norm = 0.;
    scalar_t ncc = 0.;
    for (int x = -patch_x_begin; x <= patch_x_end; x++) {
    for (int y = -patch_y_begin; y <= patch_y_end; y++) {
        scalar_t curr_src = left_img[left_x + x][left_y + y];
        scalar_t curr_dest = right_img[right_x + x][right_y + y];
        ncc += curr_src * curr_dest;
        left_norm += curr_src * curr_src;
        right_norm += curr_dest * curr_dest;
    }}
    left_norm = sqrt(left_norm);
    right_norm = sqrt(right_norm);
    ncc /= (left_norm * right_norm);
    return ncc;
}




template <typename scalar_t>
__global__
void k_disparity_map(const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> left_img,
                     const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> right_img,
                     const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> f,
                     int patch_rad,
                     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dis_map,
                     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> accuracy_map,
                     torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> dense_matches) {
       // Load coordinates
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        const int r = left_img.size(0);
        const int c = left_img.size(1);
        if (i < r && j < c) {
            // Line equation calculation. Image coordinates are inverted
            const float right_line_a = f[0][0] * j + f[0][1] * i + f[0][2];
            const float right_line_b = f[1][0] * j + f[1][1] * i + f[1][2];
            const float right_line_c = f[2][0] * j + f[2][1] * i + f[2][2];
            // Loop invariants
            scalar_t best_match_score = 0.;
            int best_match_x = 0;
            int best_match_y = 0;
            // image coordinates are inverted
            for (int y = 0; y < r; y++) {
                int x = static_cast<int>(-(right_line_a * y + right_line_c) / right_line_b);
                if (x >= 0 && x < c) {
                    scalar_t curr_match_score = local_patch_ncc(left_img, right_img, i, j, x, y, patch_rad);
                    if (best_match_score < curr_match_score) {
                        best_match_score = curr_match_score;
                        best_match_x = x;
                        best_match_y = y;
                    }
                }
            }
            // Again, invert to reflect 
            dis_map[i][j][0] = best_match_y - j;
            dis_map[i][j][1] = best_match_x - i;
            accuracy_map[i][j] = best_match_score;
            dense_matches[i][j][0] = best_match_x;
            dense_matches[i][j][1] = best_match_y;
        }
}


std::vector<torch::Tensor> disparity_map(const torch::Tensor& left_img,
                                         const torch::Tensor& right_img,
                                         const torch::Tensor& f,
                                         int patch_rad) {
    const int h = left_img.size(0);
    const int w = left_img.size(1);
    const dim3 blocks((h + BLOCK_H - 1) / BLOCK_H, (w + BLOCK_W - 1) / BLOCK_W);
    const dim3 threads(BLOCK_H, BLOCK_W);
    auto dis_map = torch::zeros({h, w, 2},
                                left_img.options());
    auto accuracy_map = torch::zeros_like(left_img);
    auto dense_matches = torch::zeros({h, w, 2},
                                      left_img.options().dtype(torch::kInt32));
    AT_DISPATCH_FLOATING_TYPES(left_img.scalar_type(), "k_disparity_map", ([&] {
        k_disparity_map<scalar_t><<<blocks, threads>>>(
                left_img.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                right_img.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                f.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                patch_rad,
                dis_map.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                accuracy_map.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                dense_matches.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>());
    }));
    CHECK_LAUNCH;
    return {dis_map, accuracy_map, dense_matches};
}
