# Makefile for compiling and running add.cu with nvcc
#
# Compiler
# Set the compiler to use for CUDA code
NVCC = nvcc

# Set the flags to pass to the compiler
NVCCFLAGS = -O3 --std=c++11

# Define the object files to be generated
CONVOLVE_OBJ = convolve.o
CLASSIFIER_OBJ = classifier.o

# Define the targets
convolve: $(CONVOLVE_OBJ)
	$(NVCC) $(NVCCFLAGS) -o convolve $(CONVOLVE_OBJ)

classifier: $(CLASSIFIER_OBJ)
	$(NVCC) $(NVCCFLAGS) -o classifier $(CLASSIFIER_OBJ)

# Define the rules for generating the object files
convolve.o: convolve.cu
	$(NVCC) $(NVCCFLAGS) -c convolve.cu

classifier.o: classifier.cu
	$(NVCC) $(NVCCFLAGS) -c classifier.cu

run_convolve: convolve 
	./convolve

run_classifier: classifier
	./classifier


# Define the clean target
clean:
	rm -f convolve classifier $(CONVOLVE_OBJ) $(CLASSIFIER_OBJ)

