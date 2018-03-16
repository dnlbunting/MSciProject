CXX=`../BOUT-dev/bin/bout-config --cxx`
CFLAGS=`../BOUT-dev/bin/bout-config --cflags`
LD=`../BOUT-dev/bin/bout-config --ld`
LDFLAGS=`../BOUT-dev/bin/bout-config --libs`

limiter: elm.cxx
	$(CXX) $(CFLAGS) -c elm.cxx -o elm.o
	$(LD) -o elm elm.o $(LDFLAGS)
