#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int stride = l.stride;
    int kernel_size = l.size;    
    int kernel_center = (l.size-1)/2;
    for (int n = 0; n < in.rows; n++) {
        for (int c = 0; c < l.channels; c++) {
            for (int h = 0; h < l.height; h += stride) {
                for (int w = 0; w < l.width; w += stride) {
                    int image_index = n * in.cols;
                    int channel_index = c * l.width * l.height;
                    int original_index = image_index + channel_index + h * l.width + w;
                    float max = in.data[original_index];
                    for (int a = 0; a < kernel_size; a++) {
                        for (int b = 0; b < kernel_size; b++) {
                            int in_row = a - kernel_center;
                            int in_col = b - kernel_center;
                            int in_index = original_index + in_row * l.width + in_col;
                            if (in_row + h >= 0 && in_row + h < l.height && in_col + w >= 0 && in_col + w < l.width) {
                                float in_value = in.data[in_index]; //gets the corresponsing value from the in matrix
                                if (in_value > max) {
                                    max = in_value;
                                }
                            } 
                        }
                    }   
                    int out_index =  (n * out.cols) + (c * outw * outh) + ((h/stride) * outw) + (w/stride);
                    out.data[out_index] = max; // set the correct location on the out matrix to the max value            
                }
            }
        }  
    }
    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int stride = l.stride;
    int kernel_size = l.size;    
    int kernel_center = (l.size)/2;
    for (int n = 0; n < in.rows; n++) {
        for (int c = 0; c < l.channels; c++) {
            for (int h = 0; h < l.height; h += stride) {
                for (int w = 0; w < l.width; w += stride) {
                    int out_index =  (n * out.cols) + (c * outw * outh) + ((h/stride) * outw) + (w/stride);
                    float max_value = out.data[out_index];
                    int max_index = 0;
                    int found = 0;
                    for (int a = 0; a < kernel_size; a++) {
                        for (int b = 0; b < kernel_size; b++) {
                            int in_row = a - kernel_center + h;
                            int in_col = b - kernel_center + w;
                            int in_index = (n * in.cols) + c * l.width * l.height + in_row * l.width + in_col;
                            if (in_row >= 0 && in_row < l.height && in_col >= 0 && in_col < l.width) {
                                float in_value = in.data[in_index]; //gets the corresponsing value from the in matrix
                                if (in_value == max_value) {
                                    max_index = in_index; // keeps track of the index for the max value
                                    found = 1;
                                    break;
                                }
                            } 
                        }
                        if (found == 1) {
                            break;
                        }
                    }   
                    prev_delta.data[max_index] += delta.data[out_index];

                }
            }
        }  
    }
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

