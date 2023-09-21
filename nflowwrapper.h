#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstddef>

class ModuleHolder;

class NFlowWrapper {
	public:
		NFlowWrapper(const char * name, std::size_t columns);
		std::vector<std::vector<float>> generate(std::vector<float> context);
		ModuleHolder * moduleHolder;
		
		std::size_t cols;
};

