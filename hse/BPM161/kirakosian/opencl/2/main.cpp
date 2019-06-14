#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <cmath>

size_t const BLOCK_SIZE = 256;

int fit_size(int n, int block_size) {
    return n % block_size == 0 ? n : (n / block_size + 1) * block_size;
}

void fill_data(std::vector<float> &v, int n) {
    float value;
    for (int i = 0; i < n; i++) {
        std::cin >> value;
        v.push_back(value);
    }
}

void write_array(std::vector<float> &v) {
    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << ' ';                 
    }
    std::cout << std::endl;
}

void call_apply_shifts(
    std::vector<float>& input,
    std::vector<float>& shifts,
    std::vector<float>& output,
    cl::Context context,
    cl::Program program,
    cl::CommandQueue queue
) {
    // allocate device buffer to hold message
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
    cl::Buffer dev_shifts(context, CL_MEM_READ_ONLY, sizeof(float) * shifts.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

    // copy from cpu to gpu
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * input.size(), &input[0]);
    queue.enqueueWriteBuffer(dev_shifts, CL_TRUE, 0, sizeof(float) * shifts.size(), &shifts[0]);
    queue.finish();
    
    // load named kernel from opencl source
    cl::Kernel kernel_hs(program, "apply_shifts");
    cl::KernelFunctor apply_shifts(
        kernel_hs, queue,
        cl::NullRange,
        cl::NDRange(fit_size(input.size(), BLOCK_SIZE)),
        cl::NDRange(BLOCK_SIZE)
    );
    cl::Event event = apply_shifts((int) input.size(), dev_input, dev_shifts, dev_output);
    event.wait();

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * output.size(), &output[0]);

}

void evaluate_prefix_sums(
    std::vector<float>& input,
    std::vector<float>& output,
    cl::Context context,
    cl::Program program,
    cl::CommandQueue queue
) {
    int n = input.size();
    
    // allocate device buffer to hold message
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * n);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

    // copy from cpu to gpu
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * n, &input[0]);

    queue.finish();

    // load named kernel from opencl source
    cl::Kernel kernel_hs(program, "scan_hillis_steele");
    cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(fit_size(n, BLOCK_SIZE)), cl::NDRange(BLOCK_SIZE));
    cl::Event event = scan_hs(
        n,
        dev_input,
        dev_output,
        cl::__local(sizeof(float) * BLOCK_SIZE),
        cl::__local(sizeof(float) * BLOCK_SIZE)
    );
    event.wait();

    if (n <= BLOCK_SIZE) {
        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * n, &output[0]);    
    } else {
        std::vector<float> result(n, 0);
        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * n, &result[0]);
        std::vector<float> sums;
        for (int i = BLOCK_SIZE - 1; i < n - 1; i += BLOCK_SIZE) {
            sums.push_back(result[i]);
        }
        evaluate_prefix_sums(sums, sums, context, program, queue);
        call_apply_shifts(result, sums, output, context, program, queue);
    }
}


int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("prefix_sum.cl");
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

        freopen("input.txt", "r", stdin);
       // freopen("output.txt", "w", stdout);

        int n;
        std::cin >> n;
        std::vector<float> input;
        std::vector<float> output(n, 0);
        fill_data(input, n);

        evaluate_prefix_sums(input, output, context, program, queue);

        write_array(output);

    }
    catch (cl::Error& e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
