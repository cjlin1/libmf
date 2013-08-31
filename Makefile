OS := $(shell uname)
CFLAGS := -O3 -pthread -std=c++0x -march=native -mtune=native -funroll-loops -Wall
SRC := src/main.cpp src/convert.cpp src/train.cpp src/predict.cpp src/view.cpp src/mf.h src/mf.cpp 

ifeq ($(OS), Linux)
	CXX = g++
else ifeq ($(OS), Darwin)
	CXX = g++-mp-4.7
	CFLAGS += -D_GLIBCXX_USE_NANOSLEEP 
endif

all: libmf

libmf: $(SRC)
	$(CXX) $(CFLAGS) -o libmf $^

clean:
	rm -f libmf
