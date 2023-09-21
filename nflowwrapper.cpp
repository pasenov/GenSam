#include "nflowwrapper.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include <torch/script.h>

class ModuleHolder {
	public:
		ModuleHolder(const char * name) : myModule(torch::jit::load(name)) {}
		torch::jit::Module myModule;
};


NFlowWrapper::NFlowWrapper(const char * name, std::size_t columns) : moduleHolder(new ModuleHolder(name)), cols(columns)   {
	moduleHolder->myModule.eval();
}


std::vector<std::vector<float>> NFlowWrapper::generate(std::vector<float> context)  {
	std::cout << "Input context size: " << context.size() << std::endl;
	
	// Validate input
	for (float f : context) {
		if (std::isnan(f)) {
			std::cout << "Error: NaN in input context!" << std::endl;
			return {};
		}
	}
	
	std::cout << "Converting context to Tensor" << std::endl;
	
	torch::Tensor t_context = torch::from_blob(context.data(), {context.size()/cols,cols});
	
	std::cout << "t_context shape: " << t_context.sizes() << std::endl;
	
	std::cout << "Generated samples tensor shape: " << moduleHolder->myModule({t_context}).toTensor().sizes() << std::endl;

	auto samples0 = moduleHolder->myModule({t_context}).toTensor().exp();
	
	std::cout << "Samples tensor value: " << samples0 << std::endl;
		
	std::cout << "Converting samples to vector" << std::endl;

	std::vector<std::vector<float>> vec;
	
	for(int i=0; i<samples0.size(0); i++) {
		std::vector<float> row;
		for(int j=0; j<samples0.size(2); j++) {
			row.push_back(samples0[i][0][j].item<float>()); 
		}
		vec.push_back(row);
	}


	// Print vector
	std::cout << "Generated samples vector: \n";
	
	for(int i=0; i<vec.size(); i++) {
		for(int j=0; j<i+samples0.size(2); j++) {
			std::cout << vec[i][j] << " "; 
		}
		std::cout << std::endl;
	}
	
	return vec;
}
