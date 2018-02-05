CXX=`../BOUT-dev/bin/bout-config --cxx`
CFLAGS=`../BOUT-dev/bin/bout-config --cflags`
LD=`../BOUT-dev/bin/bout-config --ld`
LDFLAGS=`../BOUT-dev/bin/bout-config --libs`

GSL_CFLAGS=`gsl-config --cflags`
GSL_LDFLAGS=`gsl-config --libs`

doubled: doubled.cxx
	$(CXX) $(CFLAGS) $(GSL_CFLAGS) -c doubled.cxx -o doubled.o
	$(LD) -o doubled doubled.o $(LDFLAGS) $(GSL_LDFLAGS)
