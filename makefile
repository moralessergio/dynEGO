# A simple makefile
CC = c++
CFLAGS    = -Wall -O3

# Main simulation program
SOURCE_DIR = .
OBJECT_DIR = obj
SOURCE_FILES = $(SOURCE_DIR)/utils.cpp $(SOURCE_DIR)/gp.cpp $(SOURCE_DIR)/movpeaks.cpp $(SOURCE_DIR)/strategy.cpp $(SOURCE_DIR)/trackMovPeak.cpp 
OBJECT_FILES  = $(SOURCE_FILES:.cpp=.o)
#DLINE := $(shell, cat main.h)

DLINE := $(shell more main.h | grep 'define D ')
D = $(word 3, $(DLINE))
EXENAME = $(join gpMovPeaks,_$(D)d)


# Additional libraries and header files
LIBDIR = -Lusr/lib
INCLDIR = -I. -I$(HOME)/include -I./include
LIBS  =  -lgsl -lgslcblas -lm

default:$(EXENAME)

clean: 
	rm -f *.o
	rm -f $(EXENAME)

all: clean default

$(EXENAME): $(OBJECT_FILES)
	$(CC) $(LIBDIR) $(OBJECT_FILES) $(LIBS) -o $(EXENAME)

utils.o: utils.cpp 
	$(CC) -c $(CFLAGS) $(INCLDIR) utils.cpp

gp.o: gp.cpp utils.cpp
	$(CC) -c $(CFLAGS) $(INCLDIR) gp.cpp 

movpeaks.o: movpeaks.c movpeaks.h
	$(CC) -c $(CFLAGS) $(INCLDIR) movpeaks.c 

strategy.o: strategy.cpp 
	$(CC) -c $(CFLAGS) $(INCLDIR) strategy.cpp 

trackMovPeak.o: utils.cpp strategy.cpp trackMovPeak.cpp
	$(CC) -c $(CFLAGS) $(INCLDIR) trackMovPeak.cpp 


