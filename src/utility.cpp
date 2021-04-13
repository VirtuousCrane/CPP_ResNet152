#include "utility.hpp"
#include <fstream>
#include <vector>
#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>

using namespace std;
using namespace dlib;

class Image_info{
	private:
		string filename_   ;
		string label_      ;
		long numeric_label_;

	public:
		Image_info(){
			filename_      = "_"     ;
			label_         = "others";
			numeric_label_ = 0       ;
		}

		Image_info(string label, string filename, long num){
			filename_      = filename;
			label_         = label   ;
			numeric_label_ = num     ;
		}

	string get_filename(){
		return filename_;
	}

	string get_label(){
		return label_;
	}

	long get_numeric_label(){
		return numeric_label_;
	}

	void set_label(string label){
		label_ = label;
	}

	void set_numeric_label(long n){
		numeric_label_ = n;
	}

	void set_filename(string& fn){
		filename_  = fn;
	}
};

rectangle random_crop(
		const matrix<rgb_pixel>& img,
		dlib::rand& rnd
		){

	/* Random Size */
	double mins = 0.466666666, maxs = 0.875;
	auto scale = mins + rnd.get_random_double()*(maxs-mins);
	auto size  = scale*std::min(img.nr(), img.nc());
	// img.nr() = img->height
	// img.nc() = img->width
	rectangle rect(size, size);

	/* Randomly moves the rectangle */
	point offset(   rnd.get_random_32bit_number() % (img.nc()-rect.width() ),
			rnd.get_random_32bit_number() % (img.nr()-rect.height())
			);
	return move_rect(rect, offset);
}

void randomly_crop_image(
	const matrix<rgb_pixel>& img,
	matrix<rgb_pixel>& crop     ,
	dlib::rand& rnd
	){
	/* Randomly Crops and Augments an image */

	auto rect = random_crop(img, rnd);
	extract_image_chip(img, chip_details(rect, chip_dims(227, 227)), crop);

	// Image Augmentation
	if(rnd.get_random_double() > 0.5){
		crop = fliplr(crop);
	}
	apply_random_color_offset(crop, rnd);
}

void randomly_crop_images(
		const matrix<rgb_pixel>& img,
		dlib::array<matrix<rgb_pixel>>& crops,
		dlib::rand& rnd,
		long num_crops
		){
	/* Randomly Crops and Augments multiple images */

	std::vector<chip_details> dets;
	// chip_details: Describes where an image chip is to be extracted from
	//		 within another image.
	for(long i=0; i<num_crops; ++i){
		auto rect = random_crop(img, rnd);
		dets.push_back(chip_details(rect, chip_dims(227, 227)));
	}

	extract_image_chips(img, dets, crops);

	/* Image Augmentation */
	for(auto&& img : crops){
		if(rnd.get_random_double() > 0.5){
			img = flipimg(img);
		}

		apply_random_color_offset(img, rnd);
	}
}

void process_image(string network_path, string ILSVRC_path){
	try{
		std::vector<string> labels;
		anet_type net;
		deserialize(network_path) >> net >> labels;

		// Make the last layer of the network be softmax
		softmax<anet_type::subnet_type> snet;
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
	}catch(std::exceptions& e){
		cout << e.what() << endl;
	}
}

vector<Image_info> get_imagenet_listing(
	const string& root_directory ,
	const string& image_path_file,
	const string& label_file
){
	ifstream label(label_file)          ;
	ifstream image_path(image_path_file);
	string   label_line, path           ;
	string   filename                   ;
	string   previous_label             ;

	vector<Image_info> results;
	long numeric_label = -1   ;

	while(getline(label, label_line) && getline(image_path, path)){
		filename = root_directory + "/" + path;
		if(!file_exists(filename)){
			cout << "File: <"  << filename << "> does not exist." << endl;
		}
		if(label_line != previous_label){
			++numeric_label;
		}

		Image_info temp(label_line, filename, numeric_label);
		results.push_back(temp);
	}

	return results;
}
