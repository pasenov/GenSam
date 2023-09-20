#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstddef>

#include <torch/script.h>

class ModuleHolder;

class NFlowWrapper {
	public:
		NFlowWrapper(std::string name, std::size_t columns);
		std::vector<float> generate(std::vector<float> context);
		ModuleHolder * moduleHolder;
		
		std::size_t cols;
};

