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
scalar_t local_patch_ncc(const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>& img_src,
                      const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>& img_dest,
                      int src_x,
                      int src_y,
                      int dest_x,
                      int dest_y,
                      int patch_rad) {
    int r = img_dest.size(0);
    int c = img_dest.size(1);
    int src_patch_x_begin = src_x - patch_rad < 0 ? src_x : patch_rad;
    int src_patch_x_end = src_x + patch_rad > r ? r - src_x : patch_rad;
    int src_patch_y_begin = src_y - patch_rad < 0 ? src_y : patch_rad;
    int src_patch_y_end = src_y + patch_rad > c ? r - src_y : patch_rad;
    int dest_patch_x_begin = dest_x - patch_rad < 0 ? dest_x : patch_rad;
    int dest_patch_x_end = dest_x + patch_rad > r ? r - dest_x : patch_rad;
    int dest_patch_y_begin = dest_y - patch_rad < 0 ? dest_y : patch_rad;
    int dest_patch_y_end = dest_y + patch_rad > c ? r - dest_y : patch_rad;

    int patch_x_begin = min(src_patch_x_begin, dest_patch_x_begin);
    int patch_x_end = min(src_patch_x_end, dest_patch_x_end);
    int patch_y_begin = min(src_patch_y_begin, dest_patch_y_begin);
    int patch_y_end = min(src_patch_y_end, dest_patch_y_end);
    scalar_t src_norm = 0.;
    scalar_t dest_norm = 0.;
    scalar_t ncc = 0.;
    for (int x = -patch_x_begin; x <= patch_x_end; x++) {
    for (int y = -patch_y_begin; y <= patch_y_end; y++) {
        scalar_t curr_src = img_src[src_x + x][src_y + y];
        scalar_t curr_dest = img_dest[dest_x + x][dest_y + y];
        ncc += curr_src * curr_dest;
        src_norm += curr_src * curr_src;
        dest_norm += curr_dest * curr_dest;
    }}
    src_norm = sqrt(src_norm);
    dest_norm = sqrt(dest_norm);
    ncc /= (src_norm * dest_norm);
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
            const float dest_line_a = f[0][0] * j + f[0][1] * i + f[0][2];
            const float dest_line_b = f[1][0] * j + f[1][1] * i + f[1][2];
            const float dest_line_c = f[2][0] * j + f[2][1] * i + f[2][2];
            // Loop invariants
            scalar_t best_match_score = 0.;
            int best_match_x = -INT32_MAX;
            int best_match_y = -INT32_MAX;
            // image coordinates are inverted
            for (int y = 0; y < r; y++) {
                int x = static_cast<int>(-(dest_line_a * y + dest_line_c) / dest_line_b);
                if (x >= 0 && x < c) {
                    scalar_t curr_match_score = local_patch_ncc(left_img, right_img, i, j, x, y, patch_rad);
                    if (best_match_score < curr_match_score) {
                        best_match_score = curr_match_score;
                        best_match_x = x;
                        best_match_y = y;
                    }
                }
            }
            dis_map[i][j][0] = best_match_x - i;
            dis_map[i][j][1] = best_match_y - j;
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
