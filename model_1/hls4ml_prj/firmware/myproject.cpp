//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input1[N_INPUT_1_1],
    result_t layer14_out[N_LAYER_12],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input1,layer14_out 
    #pragma HLS PIPELINE 

    const_size_in_1 = N_INPUT_1_1;
    const_size_out_1 = N_LAYER_12;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 1024>(w3, "w3.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b3, "b3.txt");
        nnet::load_weights_from_txt<model_default_t, 2048>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b6, "b6.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w9, "w9.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b9, "b9.txt");
        nnet::load_weights_from_txt<model_default_t, 160>(w12, "w12.txt");
        nnet::load_weights_from_txt<model_default_t, 5>(b12, "b12.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    input2_t fc1_input[N_INPUT_1_2];
    #pragma HLS ARRAY_PARTITION variable=fc1_input complete dim=0
    layer3_t layer3_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::dense_latency<input2_t, layer3_t, config3>(fc1_input, layer3_out, w3, b3);

    layer5_t layer5_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer3_t, layer5_t, relu_config5>(layer3_out, layer5_out);

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense_latency<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);

    layer8_t layer8_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::relu<layer6_t, layer8_t, relu_config8>(layer6_out, layer8_out);

    layer9_t layer9_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::dense_latency<layer8_t, layer9_t, config9>(layer8_out, layer9_out, w9, b9);

    layer11_t layer11_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<layer9_t, layer11_t, relu_config11>(layer9_out, layer11_out);

    layer12_t layer12_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::dense_latency<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12);

    nnet::softmax<layer12_t, result_t, softmax_config14>(layer12_out, layer14_out);

}
