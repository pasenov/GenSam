#include "nflowwrapper.h"

#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include <torch/script.h>

class ModuleHolder {
	public:
		ModuleHolder(std::string name) : myModule(torch::jit::load(name)) {}
		torch::jit::Module myModule;
};


NFlowWrapper::NFlowWrapper(std::string name, std::size_t columns) : moduleHolder(new ModuleHolder(name)), cols(columns)   {
	moduleHolder->myModule.eval();
}


std::vector<float> NFlowWrapper::generate(std::vector<float> context)  {
	std::cout << "Input context size: " << context.size() << std::endl;
	
	// Validate input
	for (float f : context) {
		if (std::isnan(f)) {
			std::cout << "Error: NaN in input context!" << std::endl;
			return {};
		}
	}
	
	std::cout << "Converting context to Tensor" << std::endl;

	/*size_t cols = static_cast<size_t>(this->cols);
	torch::Tensor t_context = torch::from_blob( context.data(), {context.size() / static_cast<size_t>(cols), cols} );*/
	
	torch::Tensor t_context = torch::from_blob(context.data(), {context.size()/cols,cols});
	
	std::cout << "t_context shape: " << t_context.sizes() << std::endl;
	
	std::cout << "Generated samples tensor shape: " << moduleHolder->myModule({t_context}).toTensor().sizes() << std::endl;

	auto samples0 = moduleHolder->myModule({t_context}).toTensor().exp();
	samples0 = samples0.squeeze(); // remove dim of size 1
	//The samples tensor had shape [1, 1, 5], so it contained 5 elements. But item() was expecting a scalar tensor, not a tensor with multiple elements. THIS MIGHT NEED TO BE CHANGED --> ANDREA?
	//Before I was getting the error:
	
/*Starting...
Input context size: 6
Converting context to Tensor
t_context shape: [1, 6]
Generated samples tensor shape: [1, 1, 5]
Samples tensor value: (1,.,.) = 
  9.7395e+04  2.1992e+00  3.3791e+00  9.9960e-01  1.0280e-01
[ CPUFloatType{1,1,5} ]
terminate called after throwing an instance of 'c10::Error'
  what():  a Tensor with 5 elements cannot be converted to Scalar


The issue is that the flow model is generating samples with 5 columns per row rather than just 1 column as expected.*/
	
	std::cout << "Samples tensor value: " << samples0 << std::endl;
	
	// Check for invalid values
	/*for (int i = 0; i < samples0.size(0); ++i) {
		for (int j = 0; j < samples0.size(1); ++j) {
			if (std::isnan(samples0[i][j].item<float>())) {
				std::cout << "Error: NaN in samples!" << std::endl;
				return {};
			}
		}
	}*/
	
	std::cout << "Converting samples to vector" << std::endl;

	
	/*std::vector<float> vec;
	for (int i = 0; i < samples0.size(0); ++i) {
		std::vector<float> row;
		for (int j = 0; j < samples0.size(1); ++j) {
			row.push_back(samples0[i][j].item<float>());
		}
		for(float f : row) {
			vec.push_back(f);
		}
	}*/
	
	std::vector<float> vec;
	for (int i = 0; i < samples0.size(0); ++i) {
		vec.push_back(samples0[i].item<float>());
	}

	// Print vector
	std::cout << "Generated samples vector: ";
	for(auto f : vec) {
		std::cout << f << " ";
	}
	std::cout << std::endl;
	
	return vec;
}

