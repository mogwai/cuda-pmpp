import torchvision.transforms.functional as tvf
from PIL import Image
import torch
from urllib import request
import io, os
from torch.utils.cpp_extension import load_inline


os.environ['CUDA_LAUNCH_BLOCKING']='1'

show_img = lambda x: tvf.to_pil_image(x)


cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

'''

def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=True):
    return load_inline(cuda_sources=[cuda_begin+cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")

def example():
    img = request.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/1024px-Dog_Breeds.jpg").read()
    image = Image.open(io.BytesIO(img))
    img = tvf.to_tensor(image)
    return img
