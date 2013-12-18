#ifndef __CPUFUNCTION
#define __CPUFUNCTION

#include <dlfcn.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include "genFunction.hpp"

class occaMemory;


class cpuHelper;

FUNCTION_POINTER_TYPEDEF;

class cpuFunction : public genFunction {

public:

  char          *name;
  char          *source;
  char          objectName[BUFSIZ];
  char          *functionName;
  voidFunction kernel;

  int  dim;
  int dims[6];

  double ev_start, ev_end;

  void load_program_source(const char *filename) {

    struct stat statbuf;
    FILE *fh = fopen(filename, "r");
    if (fh == 0){
      printf("Failed to open: %s\n", filename);
      throw 1;
    }

    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

  }


public:

  cpuFunction(){
  }

  cpuFunction(const string sourcefilename, string functionname, const string flags){

    int err;

    char cmd[BUFSIZ];
    sprintf(objectName, ".occa/%s_%d.so", functionname.c_str(), getpid());

#if 0
    sprintf(cmd, "g++ %s -m64 -fopenmp -I. -x c++ -w -fPIC -shared %s -E -ldl",
	    flags.c_str(), sourcefilename.c_str());
    system(cmd);
#endif

    sprintf(cmd, "g++ %s -m64 "
	    " -g -fopenmp -I. -x c++ -w -fPIC -shared %s -o %s  -ldl "
	    " -O3 -ftree-vectorizer-verbose=4 -mtune=native -ftree-vectorize -funroll-loops -fsplit-ivs-in-unroller -ffast-math",
	    flags.c_str(), sourcefilename.c_str(), objectName);

    cout << cmd << endl;

    int systemReturn = system(cmd);
    if(systemReturn)
      cout << systemReturn;

    functionName = strdup(functionname.c_str());

    void *obj = dlopen(objectName, RTLD_NOW);
    kernel    = (voidFunction) dlsym(obj, functionName);
  }


  void setThreadArray(size_t *in_global, size_t *in_local, int in_dim){

    dim = in_dim;

    dims[0] = in_local[0];
    dims[1] = in_local[1];
    dims[2] = in_local[2];

    dims[3] = in_global[0]/in_local[0];
    dims[4] = in_global[1]/in_local[1];
    dims[5] = 1;

  }

  CPU_KERNEL_OPERATORS;

  void tic(){
    ev_start = MPI_Wtime();
  }


  void toc(){
    ev_end =  MPI_Wtime();
  }

  double elapsedTime()
  {
    return ev_end-ev_start;
  }

};
#endif
