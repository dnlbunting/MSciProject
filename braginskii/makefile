CXX=`../BOUT-dev/bin/bout-config --cxx`
CFLAGS=`../BOUT-dev/bin/bout-config --cflags`
LD=`../BOUT-dev/bin/bout-config --ld`
LDFLAGS=`../BOUT-dev/bin/bout-config --libs`

braginskii: braginskii.cxx
	$(CXX) $(CFLAGS) -c braginskii.cxx -o braginskii.o
	$(LD) -o braginskii braginskii.o $(LDFLAGS)
