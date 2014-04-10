.SUFFIXES: .o .cpp .h
CXX = g++
CFLAGS := -O3 -pthread -std=c++0x -march=native -funroll-loops -Wall -Wl,--no-as-needed
OBJ = mf.o convert.o train.o predict.o view.o
#DFLAG = -DNOSSE
#DFLAG = -DUSEAVX
#CFLAGS += -mavx

all: libmf

%.o: src/%.cpp src/mf.h
	$(CXX) $(CFLAGS) $(DFLAG) -c -o $@ $<

libmf: $(OBJ) src/main.cpp src/mf.h
	$(CXX) $(CFLAGS) -o libmf $^

clean:
	rm -f $(OBJ) libmf
