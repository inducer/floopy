#ifndef __GENFUNCTION
#define __GENFUNCTION 1

#define MAX_GLOBAL_SIZE 1000000000
#define MAX_LOCAL_SIZE 1024

class genFunction {
public:

  int  dim;
  size_t local[3];
  size_t global[3];

  genFunction(){
  }

  virtual void setThreadArray(size_t *in_global, size_t *in_local, int in_dim) = 0;

  GEN_KERNEL_OPERATORS;

};
#endif
