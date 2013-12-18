# Settings
DATAFLOAT      = float
PROJECT_RENDER = 1
VERBOSE_OUTPUT = 1
SCREENSHOT     = 0
HAS_SOLUTION   = 0

OCCA_USE_OPENCL = 1
OCCA_USE_CUDA   = 1
OCCA_USE_CPU    = 1
OCCA_USE_ALL    = 0


# Directories and flags
SDIR   = src
IDIR   = include
ODIR   = obj

OCCADIR = libocca
OCCAOBJDIR = libocca

SOURCES  = $(wildcard $(SDIR)/*cpp)
INCLUDES = $(wildcard $(IDIR)/*hpp) $(wildcard $(IDIR)/*tpp)
OBJECTS  = $(subst $(SDIR),$(ODIR),$(SOURCES:.cpp=.o))

OCCASOURCES  = $(wildcard $(OCCADIR)/*cpp)
OCCAINCLUDES = $(wildcard $(OOCADIR)/*hpp)
OCCAOBJECTS  = $(subst $(OCCADIR),$(OCCAOBJDIR),$(OCCASOURCES:.cpp=.o))

CFLAGS = -I./$(OCCADIR) -I./include -I/opt/AMDAPP/include -I/usr/local/cuda/include -m64
LIBS   = -L. -L./lib -L/usr/lib -L/usr/local/cuda/lib -llapack -lblas -lm -ldl

# add models
CFLAGS += -DOCCA_USE_OPENCL=$(OCCA_USE_OPENCL) -DOCCA_USE_CUDA=$(OCCA_USE_CUDA) -DOCCA_USE_CPU=$(OCCA_USE_CPU) -DOCCA_USE_ALL=$(OCCA_USE_ALL)


# OSX
ifeq ($(shell uname -s),Darwin)
#LIBS   += -framework CUDA -framework Accelerate -framework OpenCL -framework OpenGL -framework GLUT
LIBS   += /Library/Frameworks/CUDA.framework/CUDA -framework Accelerate -framework OpenCL -framework OpenGL -framework GLUT
CFLAGS += -D 'OS_OSX'
# CC = clang++ -std=c++11 -stdlib=libc++ -Qunused-arguments
CC = mpic++

# LINUX
else
LIBS   += -lOpenCL -lcuda -lglut -lGL -lGLU -lGLEW
CFLAGS += -D 'OS_LINUX'
CC = mpic++
# .mpich2
endif

# Debug Option
ifeq ($(DEBUG), 1)
CFLAGS += -g
else
# <> For debugging purposes
CFLAGS += -g -rdynamic -O3
endif

# datafloat
CFLAGS += -Ddatafloat=$(DATAFLOAT)

# Verbose output
ifneq ($(VERBOSE_OUTPUT), 0)
CFLAGS += -DVERBOSE_OUTPUT
endif

# Render mode
ifneq ($(PROJECT_RENDER), 0)
CFLAGS += -DPROJECT_RENDER
endif

# Screenshot mode
ifneq ($(SCREENSHOT), 0)
CFLAGS += -DSCREENSHOT
endif

brain: main.cpp $(OBJECTS) $(INCLUDES) $(OCCAOBJECTS) $(OCCAINCLUDES)
	$(CC) $(CFLAGS) $(OBJECTS) $(OCCAOBJECTS) main.cpp -o main $(LIBS)

$(ODIR)/%.o:$(SDIR)/%.cpp $(wildcard $(subst $(SDIR),$(IDIR),$(<:.cpp=.hpp)))
	$(CC) $(CFLAGS) -c $< -o $(subst $(SDIR),$(ODIR),$(<:.cpp=.o))


$(OCCAOBJDIR)/%.o:$(OCCADIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $(subst $(OCCADIR),$(OCCAOBJDIR),$(<:.cpp=.o))

clean:
	rm -f $(ODIR)/*
	rm -f $(OCCAOBJDIR)/*.o
	rm -f main
