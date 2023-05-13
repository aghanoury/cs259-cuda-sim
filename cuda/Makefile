C=nvcc

OPT?=-O3
CFLAGS= --std=c++11 -g $(OPT) 

conv: convolution.cu
	$(C) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3 -DKy=3 -DNi=64 -DNn=64 -DTii=32 -DTi=16 -DTnn=32 -DTn=16 -DTx=7 -DTy=7

conv2: convolution.cu
	$(C) $^ $(CFLAGS) -o $@ -DNx=14 -DNy=14 -DKx=3 -DKy=3 -DNi=512 -DNn=512 -DTii=32 -DTi=16 -DTnn=32 -DTn=16 -DTx=7 -DTy=7

clas: classifier.cu
	$(C) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3 -DKy=3 -DNi=25088 -DNn=4096 -DTii=32 -DTi=16 -DTnn=32 -DTn=16 -DTx=7 -DTy=7

clas2: classifier.cu
	$(C) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3 -DKy=3 -DNi=4096 -DNn=1024 -DTii=32 -DTi=16 -DTnn=32 -DTn=16 -DTx=7 -DTy=7