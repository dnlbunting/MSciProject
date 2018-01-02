CXX=`../BOUT-dev/bin/bout-config --cxx`
CFLAGS=`../BOUT-dev/bin/bout-config --cflags`
LD=`../BOUT-dev/bin/bout-config --ld`
LDFLAGS=`../BOUT-dev/bin/bout-config --libs`

conduction: conduction.cxx
	$(CXX) $(CFLAGS) -c conduction.cxx -o conduction.o
	$(LD) -o conduction conduction.o $(LDFLAGS)
