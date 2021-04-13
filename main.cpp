#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include "src/utility.hpp"

using namespace std;
using namespace dlib;
using namespace utility;

/*
=========================================================================
 Basic Theory of Residual Networks
=========================================================================
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
=========================================================================
 The Architecture that will be used in this project
=========================================================================
									
=========================================================================
 ResNet 152
=========================================================================
 [7x7, 64, stride 2     ]
 [3x3 max pool, stride 2]

 [Bottleneck 64, 256  ] x3
 [Bottleneck 128, 512 ] x8
 [Bottleneck 256, 1024] x36
 [Bottleneck 512, 2048] x3

 [average pool]
 [1000-d fc   ]
 [softmax     ]
=========================================================================
*/

// First Layer
template<
	template <typename> class BN         ,
	template <typename> class ACTIVATION,
	typename SUBNET
	>
using custom_ResNet_input_layer = ACTIVATION<BN<con<64, 7, 7, 2, 2, SUBNET>>>;

template<typename SUBNET> using relu_input_layer = custom_ResNet_input_layer<bn_con, relu, SUBNET>;

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

template<typename SUBNET> using block_64  = relu_block<64 , SUBNET>;
template<typename SUBNET> using block_128 = relu_block<128, SUBNET>;
template<typename SUBNET> using block_256 = relu_block<256, SUBNET>;
template<typename SUBNET> using block_512 = relu_block<512, SUBNET>;

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

template<typename SUBNET> using downsampling_128 = relu_downsampling<128, SUBNET>;
template<typename SUBNET> using downsampling_256 = relu_downsampling<256, SUBNET>;
template<typename SUBNET> using downsampling_512 = relu_downsampling<512, SUBNET>;

using net_type = loss_multiclass_log<
			fc<1000,
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
			input_rgb_image_sized<227>
			>>>>>>>>>>>>>;

// ======================================================================

int main(int argc, char** argv) try{
	if (argc == 1){
		cout << "Please enter path to dataset as cmd line argument";
		return 1;
	}

	auto listing = get_imagenet_listing(string(argv[1]), string(argv[2]), string(argv[3]));
	cout << "No. of image in dataset: " << listing.size() << endl;
	const auto number_of_classes = listing.back().get_numeric_label()+1;

	set_dnn_prefer_smallest_algorithms();

	const double initial_learning_rate = 0.1   ;
	const double weight_decay          = 0.0001;
	const double momentum              = 0.9   ;

	net_type net;
	dnn_trainer<net_type> trainer(net, sgd(weight_decay, momentum));
	trainer.be_verbose();
	trainer.set_learning_rate(initial_learning_rate);
	trainer.set_synchronization_file("./src/sync/ResNet152.dat", std::chrono::minutes(10));

	trainer.set_iterations_without_progress_threshold(1000);
	set_all_bn_running_stats_window_sizes(net, 50);

	std::vector<matrix<rgb_pixel>> samples;
	std::vector<unsigned long>     labels;

	// Start threads that read images from disk and pull out
	// random crops.
	// This keeps the GPU busy.

	dlib::pipe<std::pair<Image_info, matrix<rgb_pixel>>> data(200);
	// - Pipe is a FIFO queue with a fixed max size (specified on
	//   creation) containing items of type T.
	// - Suitable for passing objects between threads.
	// - Is Thread Safe

	auto f = [&data, &listing](time_t seed){
		dlib::rand rnd(time(0)+seed);
		matrix<rgb_pixel> img;
		std::pair<Image_info, matrix<rgb_pixel>> temp;
		while(data.is_enabled()){
			temp.first = listing[rnd.get_random_32bit_number() % listing.size()];
			load_image(img, temp.first.get_filename());
			utility::randomly_crop_image(img, temp.second, rnd);
			data.enqueue(temp);
		}
	};
	std::thread data_loader1([f](){f(1);});
	std::thread data_loader2([f](){f(2);});
	std::thread data_loader3([f](){f(3);});
	std::thread data_loader4([f](){f(4);});

	while(trainer.get_learning_rate() >= initial_learning_rate * 1e-3){
		samples.clear();
		labels.clear();

		std::pair<Image_info, matrix<rgb_pixel>> img;
		while(samples.size() < 160){
			data.dequeue(img);

			samples.push_back(std::move(img.second));
			labels.push_back(img.first.get_numeric_label());
		}

		trainer.train_one_step(samples, labels);
	}

	data.disable();
	data_loader1.join();
	data_loader2.join();
	data_loader3.join();
	data_loader4.join();

	trainer.get_net();
	cout << "Saving Network" << endl;
	serialize("src/ResNet152.dnn") << net;
} catch(std::exception& e){
	cout << e.what() << endl;
}
