#ifndef __CPUHELPER
#define __CPUHELPER

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <iostream>
using namespace std;

#define ulNULL ((unsigned long int * ) NULL)

// void cl_build_kernel(cl_context context, cl_device_id device,
// 		     string filename, string flags, cl_kernel *kernel);

#include "cpuFunction.hpp"

class cpuHelper {

private:

  /* set up CPU */
  long int memUsage;

public:

  cpuHelper(int plat, int dev){
    memUsage = 0;
  }

  cpuFunction buildKernel(const string sourcefilename, const string functionname, string defines = "", string flags = ""){
    if(defines != ""){
      int start = 0;

      for(int i = 0; i < sourcefilename.length(); i++)
        if(sourcefilename[i] == '/')
          start = (i+1);

      int length = sourcefilename.length() - start;
      string source = sourcefilename.substr(start, length);

      char defname[BUFSIZ];
      sprintf(defname, ".occa/%s", source.c_str());

      ofstream defs(defname);

      defs << defines << "\n";

      ifstream sourceContents(sourcefilename.c_str());

      defs << string(std::istreambuf_iterator<char>(sourceContents), std::istreambuf_iterator<char>());

      defs.close();
      sourceContents.close();

      cout << "building: " << functionname << " from " <<  defname << endl;
      return cpuFunction(defname, functionname.c_str(), flags.c_str());
    }
    else{
      cout << "building: " << functionname << " from " <<  sourcefilename << endl;
      return cpuFunction(sourcefilename, functionname, flags);
    }
  }

  void *createBuffer(size_t sz, void *data){

    memUsage += sz;

    void *buf = (void*) malloc(sz);
    //    void *buf;
    //    posix_memalign(&buf, 16, sz);
    if(data != NULL)
      memcpy(buf, data, sz);

    return buf;
  }


  long int reportMemoryUsage(){

    return memUsage;

  }

  void finish(){}

  void flush(){}

  void destroyBuffer(void *a){

    if(a)
      free(a);
  }


  void toHost(size_t sz, void *dest, void *source){

    memcpy(dest, source, sz);

  }

  void toHost(size_t offset, size_t sz, void *dest, void *source){

    memcpy(dest, (void*) ((char*) source + offset), sz);

  }

  void toDevice(size_t sz, void *dest, void *source){

    memcpy(dest, source, sz);

  }


  void queueMarker(double &ev){
    ev = MPI_Wtime();
  }


  double elapsedTime(double ev_start, double ev_end)
  {
    return (ev_end - ev_start);
  }

  // For now just 1
  int preferredWorkgroupMultiple(){
    return 1;
  }

};


#endif
