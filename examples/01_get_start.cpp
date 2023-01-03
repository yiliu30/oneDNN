/*******************************
* Getting started with oneDNN
* 2023-1-3
* Author: Ray
* cmd: 
*   g++ -I ${DNNLROOT}/include -L ${DNNLROOT}/lib64 get_start.cpp -ldnnl -o user_start.o
*   ./user_start.o
********************************/

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_debug.h"

#include "example_utils.hpp"

using namespace dnnl;

void getting_started_tutorial(engine::kind engine_kind){
    // Initialize engine
    engine eng(engine_kind, 0);

    // Initialize stream
    stream engine_stream(eng);

    // Create user' data
    const int N = 1, H = 13, W = 13, C = 3;
    const int stride_N = H * W * C;
    const int stride_H = W * C;
    const int stride_W = C;
    const int stride_C = 1;

    // Map logical index to physical offset
    auto offset = [=](int n, int h, int w, int c) {
        return n * stride_N + h * stride_H + w * stride_W + stride_C;
    };

    // Image size
    const int image_size = N * H * W * C;

    // Allocate a buffer for the image
    std::vector<float> image(image_size);

    // Initialize the image with some values
    for (int n = 0; n < N; ++n){
        for (int h = 0; h < H; ++h){
            for (int w = 0; w < W; ++w){
                for (int c = 0; c < C; ++c){
                    int off = offset(n, h, w, c);
                    image[off] = -std::cos(off / 10.f);
                }
            }
        }
    }

    // Wrap image in dnnl:memory object
    // 1. Initializing the memory struct, no data
    //      tensor dimensions
    //      data type
    //      memory format tag
    // 2. Creating obj itself
    auto src_md = memory::desc(
        {N, C, H, W}, // logical dims
        memory::data_type::f32, // tensor's data type
        memory::format_tag::nhwc // memory format, NHWC 
    );

    // Alternative way create a memory descriptor
    auto alt_src_md = memory::desc(
        {N, C, H, W}, // logical dims
        memory::data_type::f32, // tensor's data type
        {stride_N, stride_C, stride_H, stride_W} // strides
    );

    if (src_md != alt_src_md){
        throw std::logic_error("Memory descriptor initialization mismatch.");
    }

    // Create memory objects for ReLU primitive
    auto src_mem = memory(src_md, eng);
    write_to_dnnl_memory(image.data(), src_mem);

    auto dst_mem = memory(src_md, eng);


    // Create ReLU primitive
    // 1. Create an op primitive descriptor
    // 2. Create a primitive 

    // descriptor
    auto relu_pd = eltwise_forward::primitive_desc(
        eng, // an engine the primitive will be created for
        prop_kind::forward_inference, algorithm::eltwise_relu,
        src_md,
        src_md,
        0.f,
        0.f
    );

    // ReLU primitive
    auto relu = eltwise_forward(relu_pd);

    // Executing
    relu.execute(engine_stream,
        {
            {DNNL_ARG_SRC, src_mem}, // source tag and memory obj
            {DNNL_ARG_DST, dst_mem}, // destination tag and memory obj
        });
    engine_stream.wait();

    // Obtaining the result and validation

    std::vector<float> relu_image(image_size);
    read_from_dnnl_memory(relu_image.data(), dst_mem);

    // check result
    for (int n = 0; n < N; ++n){
        for (int h = 0; h < H; ++h){
            for (int w = 0; w < W; ++w){
                for (int c = 0; c < C; ++c){
                    int off = offset(n, h, w, c);
                    float expected = image[off] < 0 ? 0.f : image[off];
                    if(relu_image[off] != expected){
                        std::cout << "At index(" << n << ", " << c << ", " << h
                                << ", " << w << ") expect " << expected
                                << " but got " << relu_image[off]
                                << std::endl;
                        throw std::logic_error("Accuracy check failed.");
                    }
                }
            }
        }
    }



}

int main(int argc, char **argv){
    int exit_code = 0;

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    try {
        getting_started_tutorial(engine_kind);
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::string &e) {
        std::cout << "Error in the example: " << e << "." << std::endl;
        exit_code = 2;
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << engine_kind2str_upper(engine_kind) << "." << std::endl;
    finalize();
    return exit_code;
}