#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include "src/utility.hpp"

using namespace std;
using namespace dlib;
using namespace utility;

const string BASE_PATH = "/content/drive/MyDrive/datasets/ILSVRC_2015/img";
const string IMG_PATH  = "../src/train_test/data_path_15.txt"             ;
const string LBL_PATH  = "../src/train_test/key_file_15.txt"              ;

const string VAL_IMG_PATH = "../src/train_test/ex_tiny_val_data_15.txt";
const string VAL_LBL_PATH = "../src/train_test/ex_tiny_val_file_15.txt";

// First Layer
template<
	template <typename> class BN         ,
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
			input_rgb_image_sized<224>
			>>>>>>>>>>>>>;

using test_net_type = loss_multiclass_log<
			fc<1000,
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
			input_rgb_image_sized<224>
			>>>>>>>>>>>>>;

// ======================================================================
void process_image(string network_path, string ILSVRC_path, int argc);
int main(int argc, char** argv) try{

	auto listing = get_imagenet_listing(BASE_PATH, IMG_PATH, LBL_PATH);
	const auto number_of_classes = listing.back().get_numeric_label()+1;

	cout << "No. of image in dataset: " << listing.size()    << endl;
	cout << "No. of classes: "          << number_of_classes << endl;

	set_dnn_prefer_smallest_algorithms();

	const double initial_learning_rate = 0.1   ;
	const double weight_decay          = 0.0001;
	const double momentum              = 0.9   ;

	net_type net;
	dnn_trainer<net_type> trainer(net, sgd(weight_decay, momentum));
	trainer.be_verbose();
	trainer.set_learning_rate(initial_learning_rate);
	trainer.set_synchronization_file("../src/sync/ResNet152_224x224_10000.dat", std::chrono::minutes(1));

	trainer.set_iterations_without_progress_threshold(20000);
	set_all_bn_running_stats_window_sizes(net, 1000);

	std::vector<matrix<rgb_pixel>> samples;
	std::vector<unsigned long>     labels;

	// Start threads that read images from disk and pull out
	// random crops.
	// This keeps the GPU busy.

	// SET MINIBATCH HERE
	dlib::pipe<std::pair<Image_info, matrix<rgb_pixel>>> data(2);

	// - Pipe is a FIFO queue with a fixed max size (specified on
	//   creation) containing items of type T.
	// - Suitable for passing objects between threads.
	// - Is Thread Safe

	auto f = [&data, &listing](time_t seed){
		dlib::rand rnd(time(0)+seed);
		matrix<rgb_pixel> img;
		std::pair<utility::Image_info, matrix<rgb_pixel>> temp;
		while(data.is_enabled()){
			temp.first = listing[rnd.get_random_32bit_number() % listing.size()];
			load_image(img, temp.first.get_filename());
			utility::randomly_crop_image(img, temp.second, rnd);
			data.enqueue(temp);
		}
	};

//	std::thread data_loader1([f](){ f(1); });
//	std::thread data_loader2([f](){ f(2); });
//	std::thread data_loader3([f](){ f(3); });
//	std::thread data_loader4([f](){ f(4); });

	dlib::rand rnd(time(0));
	while(trainer.get_learning_rate() >= initial_learning_rate * 1e-3){
		samples.clear();
		labels.clear();

//		std::pair<Image_info, matrix<rgb_pixel>> img;
		// SET MINIBATCH HERE
		while(samples.size() < 2){
			matrix<rgb_pixel> img                                                    ;
			Image_info temp = listing[rnd.get_random_32bit_number() % listing.size()];
      while(true){
        try{
			    load_image(img, temp.get_filename())                                     ;
          break;
        }catch(...){
          continue;
        }
      }
			utility::randomly_crop_image(img, img, rnd);
//			data.dequeue(img);

//			samples.push_back(std::move(img.second));
//			labels.push_back(img.first.get_numeric_label());
			samples.push_back(img);
			labels.push_back(temp.get_numeric_label());
		}

		trainer.train_one_step(samples, labels);
	}

//	data.disable();
//	data_loader1.join();
//	data_loader2.join();
//	data_loader3.join();
//	data_loader4.join();

	trainer.get_net();
	cout << "Saving Network" << endl;
	serialize("../src/ResNet152_224x224_10000.dnn") << net;

// ===========================================================================
//  Testing The Neural Network
// ===========================================================================
/*
  set_dnn_prefer_smallest_algorithms();

  dlib::rand rnd(time(0));
  net_type net;
	deserialize("../src/ResNet152.dnn") >> net;
	softmax<test_net_type::subnet_type> snet;
	snet.subnet() = net.subnet()       ;

	cout << "Testing the NN" << endl;

	int num_right = 0     ;
	int num_wrong = 0     ;
	int num_right_top1 = 0;
	int num_wrong_top1 = 0;

  auto val_listing = get_imagenet_listing(BASE_PATH, VAL_IMG_PATH, VAL_LBL_PATH);
  cout << "No. of image in dataset: " << val_listing.size()                       << endl;
	cout << "No. of classes: "          << val_listing.back().get_numeric_label()+1 << endl;

	for(auto l: val_listing){
		dlib::array<matrix<rgb_pixel>> images;
		matrix<rgb_pixel>              img   ;

    while(true){
      try{
		    load_image(img, l.get_filename());
        break;
      }catch (...){
        continue;
      }
    }

    cout << "Loaded Images" << endl;

		const int num_crops = 4;
		randomly_crop_images(img, images, rnd, num_crops);

    cout << "Cropped Images" << endl;

		matrix<float, 1, 1000> p = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;

    cout << "Generated Predictions" << endl;

		// Top-1
		if(index_of_max(p) == l.get_numeric_label()){
			++num_right_top1;
		}else{
			++num_wrong_top1;
		}

		// Top-5
		bool found_match = false;
		for(int k=0; k<5; ++k){
			long predicted_label = index_of_max(p);
			p(predicted_label) = 0;
			if(predicted_label == l.get_numeric_label()){
				found_match = true;
				break;
			}
		}
		if(found_match){
			++num_right;
		}else{
			++num_wrong;
		}
	}

	cout << "Top-5 acc: " << num_right/(double)(num_right+num_wrong)                << endl;
	cout << "Top-1 acc: " << num_right_top1/(double)(num_right_top1+num_wrong_top1) << endl;
*/
} catch(std::exception& e){
	cout << e.what() << endl;
}

void process_image(string network_path, string ILSVRC_path, int argc){
	try{
		std::vector<string> labels;
		net_type net;
		deserialize(network_path) >> net >> labels;

		// Make the last layer of the network be softmax
		softmax<net_type::subnet_type> snet;
		snet.subnet() = net.subnet()        ;

		dlib::array<matrix<rgb_pixel>> images;
		matrix<rgb_pixel> img, crop          ;

		dlib::rand rnd  ;
		image_window win;

		for(int i=0; i<argc; ++i){
			load_image(img, ILSVRC_path);
			const int num_crops = 16;

			randomly_crop_images(img, images, rnd, num_crops);

			matrix<float, 1, 1000> p = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;

			win.set_image(img);

			// Getting the top-5 probabilities
			for(int k=0; k<5; ++k){
				unsigned long predicted_label = index_of_max(p);
				cout << p(predicted_label) << ": " << labels[predicted_label] << endl;
				p(predicted_label) = 0;
			}

			cout << "Hit enter to process the next image";
			cin.get();
		}
	}catch(std::exception& e){
		cout << e.what() << endl;
	}
}
