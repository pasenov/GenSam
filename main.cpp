#include "nflowwrapper.h"

#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include <torch/script.h>

int main()
{
	NFlowWrapper flow("../flow_model.pt",6);
	std::cout << "Starting..." << std::endl;

	std::vector<float> test = {30.f,2.f,14.f,5.f,6.f,12.f};
	auto res=flow.generate(test);
	//std::cout << "\n samples = \n" << res << std::endl;
	
	
	if (res.empty()) {
		std::cout << "Error generating samples!" << std::endl;
		return 1;
	}
	
	std::cout << "Finished successfully!" << std::endl;
	
	return 0;
}
