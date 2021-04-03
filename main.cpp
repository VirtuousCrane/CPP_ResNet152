#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;

/* The ResNet is created based on my comprehension of the paper "Deep Residual Learning From Image Recognition" */

/*
===============================================================================================================
 Basic Theory of Residual Networks
===============================================================================================================
 The Residual Network is introduced to address the degradation problem of Deep Neural Networks.

 The theory of Residual Networks is based on the formula:
	F(x) := H(x) - x
     WHERE
	F(x) = Residual Function
	H(x) = Desired Mapping

 This theory hypothesize that it is easier to reach the desired mapping by calculating from the
 Residual Function than to try and stack it randomly.

 Therefore,
	H(x) := F(x) + x
===============================================================================================================
*/

// ResNet building block with Bottleneck architecture
template<
	int N                               , // No. of Filter Maps
	template <typename> class BN        , // Batch Normalization Layer
	template <typename> class ACTIVATION, // Activation Function Layer
	int stride                          ,
	typename SUBNET                       // Input
	>
using residual_block_not_activated =    add_prev1<
					BN<
					con<N, 1, 1, stride, stride,

					ACTIVATION<
					BN<
					con<N, 3, 3, stride, stride,

					ACTIVATION<
					BN<
					con<N, 1, 1, stride, stride,
					tag1<SUBNET>>
					>>>>>>>>;

template<int N, typename SUBNET> using relu_block  = relu<residual_block_not_activated<N, bn_con, relu, 1, SUBNET>>;
template<int N, typename SUBNET> using prelu_block = prelu<residual_block_not_activated<N, bn_con, prelu, 1, SUBNET>>;

// Resnet building block for downsampling
template<
	int N,
	template <typename> class BN,
	template <typename> class ACTIVATION,
	typename SUBNET
	>
using residual_downsampling_not_activated = 	add_prev1<
						BN<
						con<N, 1, 1, 1, 1,

						ACTIVATION<
						BN<
						con<N, 3, 3, 2, 2,

						ACTIVATION<
						skip1<
						tag1<
						BN<
						con<N, 1, 1, 1, 1, SUBNET>
						>>>>>>>>>>;

template<int N, typename SUBNET> using relu_downsampling  = relu<residual_downsampling_not_activated<N, bn_con, relu, SUBNET>>;
template<int N, typename SUBNET> using prelu_downsampling = prelu<residual_downsampling_not_activated<N, bn_con, prelu, SUBNET>>;

int main(int argc, char** argv) try{

} catch(std::exception& e){
	cout << e.what() << endl;
}
