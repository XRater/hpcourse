#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <cmath>


void fill_data(std::vector<float> &v, int n) {
    float value;
    for (int i = 0; i < n * n; i++) {
        std::cin >> value;
        v.push_back(value);
    }
}

void write_array(std::vector<float> &v, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << v[i * n + j] << ' ';                 
        }
        std::cout << std::endl;
    }
}

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file),
                              (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
		try
		{
			program.build(devices);
		}
		catch (cl::Error const & e)
		{			
			std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
			std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
			std::cout << log_str;
			return 0;
		}
        size_t const block_size = 16;

        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);

        int n, m;
        std::cin >> n >> m;
        std::vector<float> a;
        std::vector<float> b;
        std::vector<float> c(n * n, 0);

        fill_data(a, n);
        fill_data(b, m);

        const size_t a_size = n * n;
        const size_t b_size = m * m;

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * a_size);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * b_size);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * a_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * a_size, &a[0]);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * b_size, &b[0]);

        // load named kernel from opencl source
        cl::Kernel kernel(program, "convolution");
        cl::KernelFunctor convolution(kernel, queue, cl::NullRange,
                                      cl::NDRange(a_size, a_size),
                                      cl::NDRange(block_size, block_size));

        convolution(dev_a, dev_b, dev_c, n, m);

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * a_size, &c[0]);

        write_array(c, n);
    }
    catch (cl::Error& e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
