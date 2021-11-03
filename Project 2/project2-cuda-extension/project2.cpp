#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// CUDA forward declarations
torch::Tensor NCC(const torch::Tensor& in, const torch::Tensor& f, bool mirror);

torch::Tensor corner_NMS(const torch::Tensor& in, const int window_rad);

torch::Tensor harris_corner_detector(const torch::Tensor& I_x,
                                     const torch::Tensor& I_y,
                                     const int width, const float k);

torch::Tensor ncc_interface(const torch::Tensor& in, const torch::Tensor& f, bool mirror) {
    CHECK_INPUT(in);
    CHECK_INPUT(f);
    return NCC(in, f, mirror);
}


torch::Tensor harris_corner_detector_interface(const torch::Tensor& I_x,
                                               const torch::Tensor& I_y,
                                               const int width, const float k) {
    CHECK_INPUT(I_x);
    CHECK_INPUT(I_y);
    assert(I_x.size(0) == I_y.size(0));
    assert(I_x.size(1) == I_y.size(1));
    return harris_corner_detector(I_x, I_y, width, k);
}



torch::Tensor corner_nms_interface(const torch::Tensor& in, const int window_rad) {
    CHECK_INPUT(in);
    return corner_NMS(in, window_rad);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ncc", &ncc_interface, "Normalized Cross Correlation");
    m.def("corner_nms", &corner_nms_interface, "Non-max suppression for corners");
    m.def("harris_corner", &harris_corner_detector_interface, "Harris corner detector");
}
