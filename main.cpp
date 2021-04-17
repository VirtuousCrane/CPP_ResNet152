#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include "src/utility.hpp"

using namespace std;
using namespace dlib;
using namespace utility;

const string       CIFAR_PATH     = "/media/akagi/Zuikaku/datasets/CIFAR-10-binary";
const string       SYNC_FILE      = "../src/sync/cifar10.dat"                      ;
const unsigned int MINIBATCH_SIZE = 64;

// ======================================================================
//  Defining the Network
// ======================================================================

// First Layer
template<
	template <typename> class BN        ,
	template <typename> class ACTIVATION,
	typename SUBNET
	>
using custom_ResNet_input_layer = ACTIVATION<BN<con<64, 7, 7, 2, 2, SUBNET>>>;

template<typename SUBNET> using relu_input_layer = custom_ResNet_input_layer<bn_con, relu, SUBNET>;
template<typename SUBNET> using arelu_input_layer = custom_ResNet_input_layer<affine, relu, SUBNET>;


// First Layer Pool
template<typename SUBNET> using input_pooling = max_pool<3, 3, 2, 2, SUBNET>;

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
					con<N*4, 1, 1, stride, stride,

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
template<int N, typename SUBNET> using arelu_block  = relu<residual_block_not_activated<N, affine, relu, 1, SUBNET>>;
template<int N, typename SUBNET> using aprelu_block = prelu<residual_block_not_activated<N, affine, prelu, 1, SUBNET>>;

template<typename SUBNET> using block_64  = relu_block<64 , SUBNET>;
template<typename SUBNET> using block_128 = relu_block<128, SUBNET>;
template<typename SUBNET> using block_256 = relu_block<256, SUBNET>;
template<typename SUBNET> using block_512 = relu_block<512, SUBNET>;

template<typename SUBNET> using ablock_64  = arelu_block<64 , SUBNET>;
template<typename SUBNET> using ablock_128 = arelu_block<128, SUBNET>;
template<typename SUBNET> using ablock_256 = arelu_block<256, SUBNET>;
template<typename SUBNET> using ablock_512 = arelu_block<512, SUBNET>;

// Resnet building block for downsampling
/*
 Main:
 1x1 Conv -> 3x3 Conv -> 1x1 Conv
 Shortcut:
 1x1 Conv

 residual_downsampling_not_activated = Main + shortcut
*/
template<
	int N                               ,
	template <typename> class BN        ,
	template <typename> class ACTIVATION,
	typename SUBNET
	>
using residual_downsampling_not_activated = 	add_prev1<
						BN<
						con<N*4, 1, 1, 1, 1,

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
template<int N, typename SUBNET> using arelu_downsampling  = relu<residual_downsampling_not_activated<N, affine, relu, SUBNET>>;
template<int N, typename SUBNET> using aprelu_downsampling = prelu<residual_downsampling_not_activated<N, affine, prelu, SUBNET>>;

template<typename SUBNET> using downsampling_128 = relu_downsampling<128, SUBNET>;
template<typename SUBNET> using downsampling_256 = relu_downsampling<256, SUBNET>;
template<typename SUBNET> using downsampling_512 = relu_downsampling<512, SUBNET>;

template<typename SUBNET> using adownsampling_128 = arelu_downsampling<128, SUBNET>;
template<typename SUBNET> using adownsampling_256 = arelu_downsampling<256, SUBNET>;
template<typename SUBNET> using adownsampling_512 = arelu_downsampling<512, SUBNET>;

using net_type = loss_multiclass_log<
			fc<10,
			avg_pool_everything<

			block_512<
			block_512<
			downsampling_512<

			repeat<35,
			block_256,
			downsampling_256<

			repeat<7,
			block_128,
			downsampling_128<

			repeat<3,
			block_64,

			input_pooling<
			relu_input_layer<
			input_rgb_image_sized<32>
			>>>>>>>>>>>>>;

using test_net_type = loss_multiclass_log<
			fc<10,
			avg_pool_everything<

			ablock_512<
			ablock_512<
			adownsampling_512<

			repeat<35,
			ablock_256,
			adownsampling_256<

			repeat<7,
			ablock_128,
			adownsampling_128<

			repeat<3,
			ablock_64,

			input_pooling<
			arelu_input_layer<
			input_rgb_image_sized<32>
			>>>>>>>>>>>>>;

// ======================================================================
int main(int argc, char** argv) try{
	std::vector<matrix<rgb_pixel>> training_images, testing_images;
	std::vector<unsigned long>     training_labels, testing_labels;

	dlib::load_cifar_10_dataset(
			CIFAR_PATH     ,
			training_images,
			training_labels,
			testing_images ,
			testing_labels
			);

	set_dnn_prefer_smallest_algorithms();

	// ==============================================================
	//  Configuring The Neural Network
	// ==============================================================

	const double initial_learning_rate = 0.1   ;
	const double weight_decay          = 0.0001;
	const double momentum              = 0.9   ;

	net_type net;
	dnn_trainer<net_type> trainer(net, sgd(weight_decay, momentum));

	trainer.be_verbose();
	trainer.set_learning_rate(initial_learning_rate);
	trainer.set_iterations_without_progress_threshold(1000);
	trainer.set_synchronization_file(
				SYNC_FILE,
				std::chrono::minutes(1)
				);

	// ==============================================================
	//  Training The Neural Network
	// ==============================================================

	std::vector<matrix<rgb_pixel>> samples;
	std::vector<unsigned long>     labels;

	dlib::rand rnd(time(0));
	while(trainer.get_learning_rate() >= initial_learning_rate * 1e-3){
		samples.clear();
		labels.clear();

		while(samples.size() < MINIBATCH_SIZE){
			auto index = rnd.get_random_32bit_number() % training_images.size();
			samples.push_back(training_images[index]);
			labels.push_back(training_labels[index]) ;
		}

		trainer.train_one_step(samples, labels);
	}

	trainer.get_net();
	cout << "Saving Network" << endl;
	serialize("../src/saved_weights/ResNet_CIFAR10.dnn") << net;

// ===========================================================================
//  Testing The Neural Network
// ===========================================================================
	deserialize("../src/ResNet152.dnn") >> net;

	softmax<test_net_type::subnet_type> snet;
	snet.subnet() = net.subnet()            ;

	cout << "Testing the NN" << endl;

	int num_right_train = 0;
	int num_wrong_train = 0;

	int num_right_test = 0;
	int num_wrong_test = 0;

	for(int i=0; i<training_images.size(); i++){
		matrix<float, 1, 10> predicted_label = sum_rows(mat(snet(training_images[i])));
		if (index_of_max(predicted_label) == training_labels[i]){
			++num_right_train;
		}else{
			++num_wrong_train;
		}
	}

	for(int i=0; i<testing_images.size(); i++){
		matrix<float, 1, 10> predicted_label = sum_rows(mat(snet(testing_images[i])));
		if(index_of_max(predicted_label) == testing_labels[i]){
			++num_right_test;
		}else{
			++num_wrong_test;
		}
	}

	cout << "Training acc: " << 100*num_right_train/(double)(num_right_train+num_wrong_train) << "%" << endl;
	cout << "Testing acc: "  << 100*num_right_test/(double)(num_right_test+num_wrong_test)    << "%" << endl;

} catch(std::exception& e){
	cout << e.what() << endl;
}

