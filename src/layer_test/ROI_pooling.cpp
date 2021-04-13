#include <dlib/dnn.h>
#include <iostream>
#include <dlib/ 

class ROI_Pooling_{
	private:
		int pooling_width_, pooling_height_;
	public:
		ROI_Pooling_();
		ROI_Pooling_(int pooling_width, int pooling_height){
			pooling_width_  = pooling_width ;
			pooling_height_ = pooling_height;
		}
		ROI_Pooling_(const ROI_Pooling_ &item);
		/* ensures that this object is copy constructable */

		ROI_Pooling_(const OTHER_LAYER_TYPE &item);
		/* ensures that this object can be constructed from another
		   layer. Is useful when you need to convert from one layer
                   to another */

		template<typename SUBNET>
		void setup(const SUBNET& sub);
		/* Performs any necessary initial memory allocations */

		template<typename SUBNET>
		void forward(
			const SUBNET& sub,
                        resizable_tensor& data_output
		){
			
		}
		/*
		  The Forward Pass
		  ========================================================
		  Requirements:
			- SUBNET has been implemented.
			- setup() has been called.
		  Ensures:
			- Runs the output of SUBNET through this layer and
			  stores the results into #data_output.
			- forward() can use any of the outputs in sub to
			  compute whatever it wants.
		*/

		template<typename SUBNET>
		void backward(
			const tensor& computed_output,
			const tensor& gradient_input,
			SUBNET& sub,
			tensor& params_grad
		);
		/*
		  Backpropagation
		  ========================================================
		  Requirements:
			- SUBNET has been implemented.
			- setup() has been called.
			- computed_output is the tensor requlting from the forward pass.
			- have_same_dimensions(gradient_input, computed_output) ==  true
			- have_same_dimensions(sub.get_gradient_input(), sub.get_output()) == true
			- have_same_dimensions(params_grad, get_layer_params()) == true

		  Ensures:
			- Outputs the gradient of this layer with respect
			  to the input data from SUBNET and this layer's
			  parameters.
			  - These gradients are stored into #sub and
			    #params_grad, respectively.
		*/

		void forward_inplace(
			const tensor& data_input,
			tensor& data_input
		);

		void backward_inplace(
			const tensor& computed_output, // Optional
			const tensor& gradient_input,
			tensor& data_grad,
			tensor& params_grad
		);

		/*
		  ========================================================
		  DO NOT IMPLEMENT forward_inplace AND backward_inplace
		  ========================================================
		*/

		const tensor& get_layer_params() const;
		/*
		  Returns the parameters that define the behavior of
		  forward().
		*/

		tensor& get_layer_params()
		/* Same */

		dpoint map_input_to_output(dpoint p) const;
		dpoint map_output_to_input(dpoint p) const;
		/* These two functions are optional
		  If provided, they should map between (column, row) coord
		  in input and output tensors of forward().
		*/

		void clean();
		/* OPTIONAL
		  ========================================================
		  - Calling clean() causes the object to forget about
		    everything except its parameters.
		    - Useful if the layer caches information between
		      forward and backward passes and you want to clean
		      the cache.
		  ========================================================
		*/
};


std::ostream& operator<<(std::ostream& out, const ROI_Pooling_ &item);
/* Prints a string describing the layer */

void to_xml(const ROI_Pooling_ &item, std::ostream& out);
/*OPTIONAL
========================================================
Prints a layer as XML
*/

void serialize(const ROI_Pooling_& item, std::ostream& out);
void deserialize(ROI_Pooling_& item, std::istream& in);

template<typename SUBNET>
using ROI_Pooling = add_layer<ROI_Pooling_, SUBNET>;
