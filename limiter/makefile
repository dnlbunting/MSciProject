CXX=`../BOUT-dev/bin/bout-config --cxx`
CFLAGS=`../BOUT-dev/bin/bout-config --cflags`
LD=`../BOUT-dev/bin/bout-config --ld`
LDFLAGS=`../BOUT-dev/bin/bout-config --libs`

limiter: limiter.cxx
	$(CXX) $(CFLAGS) -c limiter.cxx -o limiter.o
	$(LD) -o limiter limiter.o $(LDFLAGS)
