#include "stdlib.h"
#include "mpi.h"
#include "occa.hpp"


int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  int platform = 1, device = 0;

  //  void cl_list_all_devices();
  //  cl_list_all_devices();

  /* grab platform and device */
  occa dev;
  
  dev.setup("OpenCL", platform, device); // options CUDA, OpenCL, CPU
  
  /* build as cl kernel */
  occaKernel volumeKernel = dev.buildKernel("foo.occa", "foo", " ");

  /* set thread array for kernel */
  int dim = 1;
  size_t outer[3] = {100,1,1};
  size_t inner[3] = {5,1,1};
  volumeKernel.setThreadArray(outer, inner, dim);

  /* build host array */
  int N  = outer[0];
  double *h_A = (double*) calloc(N, sizeof(double));

  /* allocate device array */
  occaMemory d_A = dev.createBuffer(N*sizeof(double), NULL);

  /* start kernel */
  double d = 12.34567;
  volumeKernel(d_A, d);

  /* copy results */
  d_A.toHost(h_A);

  /* output results */
  for(int n=0;n<N;++n)
    cout << "h_A[" << n << "] = " << h_A[n] << endl;

  MPI_Finalize();

}
