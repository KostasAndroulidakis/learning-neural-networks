CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2

SRCS = Perceptron.cpp
MAIN_SRC = main.cpp
AND_GATE_SRC = and_gate.cpp

OBJS = $(SRCS:.cpp=.o)
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)
AND_GATE_OBJ = $(AND_GATE_SRC:.cpp=.o)

TARGETS = perceptron and_gate

.PHONY: all clean

all: $(TARGETS)

perceptron: $(OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

and_gate: $(OBJS) $(AND_GATE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(MAIN_OBJ) $(AND_GATE_OBJ) $(TARGETS)
