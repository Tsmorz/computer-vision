#include <torch/extension.h>
#include <iostream>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// CUDA forward declarations

std::vector<torch::Tensor> disparity_map(const torch::Tensor& right_img,
                                                                      const torch::Tensor& left_img,
                                                                      const torch::Tensor& f,
                                                                      int patch_rad);

std::vector<torch::Tensor> disparity_map_interface(const torch::Tensor& right_img,
                                                   const torch::Tensor& left_img,
                                                   const torch::Tensor& f,
                                                   int patch_rad) {
    assert(images.size() == 2);
    CHECK_INPUT(right_img);
    CHECK_INPUT(left_img);
    CHECK_INPUT(f);
    return disparity_map(right_img, left_img, f, patch_rad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("disparity_map", &disparity_map_interface, "Disparity map calculator");
}
