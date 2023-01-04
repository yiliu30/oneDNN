/*******************************
* 
* Memory format propagation
* Built around a CNN + pooling 
*
* 2023-1-4
* Author: Ray
* cmd: 
*   g++ -I ${DNNLROOT}/include -L ${DNNLROOT}/lib64 02_memory_format_propagation.cpp -ldnnl -o 02_memory_format_propagation.o
*   ./02_memory_format_propagation.o
********************************/


// Steps
// 1. Create a pooling primitive descriptor.
// 2. Create a memory descriptors for input and output data in the NCHW memory format.

#include <iostream>
#include <sstream>
#include <string>

#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

void memory_format_propagation_tutorial(engine::kind engine_kind){
    // Initialize engine and stream
    engine eng(engine_kind, 0);
    stream s(eng);

    // Create conv and pool primitives
    // A primitive should pick an optimized format for the specified computation parameters
    // 3x3 kernel, padding = 1, activation tensor shape do not change.
    const int N = 1, H = 14, W = 14, IC = 128, OC = 256, KH = 3, KW = 3;

    // Create memory descriptors.
    auto conv_src_md = memory::desc(
        {N, IC, H, W},
        memory::data_type::f32,
        memory::format_tag::any  // let convolution choose memory format
    );

    auto conv_weights_md = memory::desc(
        {OC, IC, KH, KW},
        memory::data_type::f32,
        memory::format_tag::any // let convolution choose memory format
    );

    auto conv_dst_md = memory::desc(
        {N, OC, H, W},
        memory::data_type::f32,
        memory::format_tag::any // let convolution choose memory format
    );

    const auto &pool_dst_md = conv_dst_md;  // shape does not change 

    // Pass the memory descriptors to primitive descriptors constructors.
    auto conv_pd = convolution_forward::primitive_desc(
        eng,
        prop_kind::forward_inference,
        algorithm::convolution_auto,
        conv_src_md, 
        conv_weights_md,
        conv_dst_md, // shape information
        {1, 1}, // strides
        {1, 1}, // left padding
        {1, 1} // right padding
    );

    auto pool_pd = pooling_forward::primitive_desc(
        eng,
        prop_kind::forward_inference,
        algorithm::pooling_max,
        conv_pd.dst_desc(),
        pool_dst_md, // shape information
        {1, 1}, {KH, KW}, // strides and kernel
        {0, 0}, // dilation
        {1, 1}, {1, 1} // left and right padding
    );

    // Create source and destination memory objects
    auto src_mem = memory({
        {N, IC, H, W},
        memory::data_type::f32,
        memory::format_tag::nchw},
        eng);
    
    auto weights_mem = memory({
        {OC, IC, KH, KW}, 
        memory::data_type::f32,
        memory::format_tag::oihw},
        eng);

    auto dst_mem = memory({
        {N, OC, H, W},
        memory::data_type::f32,
        memory::format_tag::nchw},
        eng);
    
    // Determine if source and destination need to be reordered
    bool need_reorder_src = conv_pd.src_desc() != src_mem.get_desc();
    bool need_reorder_weights = conv_pd.weights_desc() != weights_mem.get_desc();
    bool need_reorder_dst = conv_pd.dst_desc() != dst_mem.get_desc();

    // Allocate intermediate buffers if necessary
    auto conv_src_mem = need_reorder_dst ? memory(conv_pd.src_desc(), eng) : src_mem;
    auto conv_weights_mem = need_reorder_weights ? memory(conv_pd.weights_desc(), eng) : weights_mem;
    auto conv_dst_mem = memory(conv_pd.dst_desc(), eng);
    auto pool_dst_mem = need_reorder_dst ? memory(pool_pd.dst_desc(), eng) : dst_mem;

    // Perform reorders for source data if necessary
    if(need_reorder_src){
        auto reorder_src = reorder(src_mem, conv_src_mem);
        reorder_src.execute(
            s,
            {
                {DNNL_ARG_FROM, src_mem},
                {DNNL_ARG_TO, conv_weights_mem}});
        s.wait(); // wait for the reorder to complete.
    }

    if(need_reorder_weights){
        auto reorder_weights = reorder(weights_mem, conv_weights_mem);
        reorder_weights.execute(s,
        {
            {DNNL_ARG_FROM, weights_mem},
            {DNNL_ARG_TO, conv_weights_mem}});
        s.wait();
    }

    // Create and execute convolution and pooling
    auto conv_scratchpad_mem = memory(conv_pd.scratchpad_desc(), eng);
    auto conv = convolution_forward(conv_pd);
    conv.execute(
        s,
        {
            {DNNL_ARG_FROM, conv_src_mem},
            {DNNL_ARG_WEIGHTS, conv_weights_mem},
            {DNNL_ARG_DST, conv_dst_mem}});
    
    auto pool_scratchpad_mem = memory(pool_pd.scratchpad_desc(), eng);
    auto pool = pooling_forward(pool_pd);
    pool.execute(
        s,
        {
            {DNNL_ARG_SRC, conv_dst_mem},
            {DNNL_ARG_DST, pool_dst_mem}});
        s.wait();
    
    // Reorder destination data if necessary
    if(need_reorder_dst){
        auto reorder_dst = reorder(pool_dst_mem, dst_mem);
        reorder_dst.execute(
            s,
            {
                {DNNL_ARG_FROM, pool_dst_mem},
                {DNNL_ARG_TO, dst_mem}});
    }
}

int main(int argc, char **argv) {
    return handle_example_errors(
            memory_format_propagation_tutorial, parse_engine_kind(argc, argv));
}

/*
onednn_verbose,info,oneDNN v2.7.0 (commit 4358e67d480bd11baca94861a8b4b28db4be894f)
onednn_verbose,info,cpu,runtime:OpenMP,nthr:48
onednn_verbose,info,cpu,isa:Intel AVX-512 with Intel DL Boost
onednn_verbose,info,gpu,runtime:none
onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,,,1x128x14x14,12.0972
onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb32a:f0,,,256x128x3x3,10.314
onednn_verbose,exec,cpu,convolution,brgconv:avx512_core,forward_inference,src_f32::blocked:acdb:f0 wei_f32::blocked:Acdb32a:f0 bia_undef::undef::f0 dst_f32::blocked:acdb:f0,,alg:convolution_direct,mb1_ic128oc256_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1,10.429
onednn_verbose,exec,cpu,pooling,jit:avx512_core,forward_inference,src_f32::blocked:acdb:f0 dst_f32::blocked:acdb:f0 ws_undef::undef::f0,,alg:pooling_max,mb1ic256_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1,10.1711
onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:acdb:f0 dst_f32::blocked:abcd:f0,,,1x256x14x14,10.0352
Example passed on CPU.

*/