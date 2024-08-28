from common import *

img = example()
show_img(img)

func_name = "blur"

cuda_src = r'''
__global__ 
void blur_kernel(float* x, float* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("i = %d\n", i);
}

torch::Tensor blur(torch::Tensor input) {
    CHECK_INPUT(input);
    int c = input.size(0);
    int h = input.size(1);
    int w = input.size(2);
    int wh = w*h;
    int threads = 256;
    auto output = torch::empty({c,h,w}, input.options());
    
    blur_kernel<<<cdiv(wh,threads), threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        wh
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}'''

func_def = f"torch::Tensor {func_name}(torch::Tensor input);"

module = load_cuda(cuda_src, func_def, [func_name], verbose=True)
out = getattr(module,func_name)(img.contiguous().cuda())
show_img(out)
