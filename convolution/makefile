CXX=`../BOUT-dev/bin/bout-config --cxx`
BOUT_CFLAGS=`../BOUT-dev/bin/bout-config --cflags`
LD=`../BOUT-dev/bin/bout-config --ld`
BOUT_LDFLAGS=`../BOUT-dev/bin/bout-config --libs`

GSL_CFLAGS=`gsl-config --cflags`
GSL_LDFLAGS=`gsl-config --libs`

convolution: convolution.cxx
	$(CXX)  $(BOUT_CFLAGS) $(GSL_CFLAGS) -c convolution.cxx -o convolution.o -std=c++14
	$(LD) -o convolution convolution.o $(BOUT_LDFLAGS) $(GSL_LDFLAGS)
