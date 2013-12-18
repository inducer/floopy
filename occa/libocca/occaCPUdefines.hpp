#ifndef __OCCACPUDEFINES
#define __OCCACPUDEFINES

#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "omp.h"

int occaDims0 = 0, occaDims1 = 0, occaDims2 = 0;

typedef struct foo1 { float  x,y; }     float2;
typedef struct foo4 { float  x,y,z,w; } float4;
typedef struct doo2 { double x,y; }     double2;
typedef struct doo4 { double x,y,z,w; } double4;

#define occaInnerDim0 (occaDims[0])
#define occaInnerDim1 (occaDims[1])
#define occaInnerDim2 (occaDims[2])

#define occaOuterDim0 (occaDims[3])
#define occaOuterDim1 (occaDims[4])
#define occaOuterDim2 (occaDims[5])

#define occaGlobalDim0 (occaInnerDim0*occaOuterDim0)
#define occaGlobalDim1 (occaInnerDim1*occaOuterDim1)
#define occaGlobalDim2 (occaInnerDim2*occaOuterDim2)

// splitting these loops with openmp is problematic because of the global loop variables
// the global loop variables are used to locate private members.
#define occaInnerFor0 for(occaInnerId0 = 0; occaInnerId0 < occaInnerDim0; ++occaInnerId0)
#define occaInnerFor1 for(occaInnerId1 = 0; occaInnerId1 < occaInnerDim1; ++occaInnerId1)
#define occaInnerFor2 for(occaInnerId2 = 0; occaInnerId2 < occaInnerDim2; ++occaInnerId2)

#define occaInnerFor occaInnerFor2 occaInnerFor1 occaInnerFor0


#define occaOuterFor0                                                   \
  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;             \
  _Pragma("omp parallel for firstprivate(occaInnerId0,occaInnerId1,occaInnerId2,occaDims0,occaDims1,occaDims2)") \
for(int occaOuterId0 = 0; occaOuterId0 < occaOuterDim0; ++occaOuterId0)

//  _Pragma("omp parallel for schedule(static,256) firstprivate(occaInnerId0,occaInnerId1,occaInnerId2,occaDims0,occaDims1,occaDims2)") \

#define occaOuterFor1 for(int occaOuterId1 = 0; occaOuterId1 < occaOuterDim1; ++occaOuterId1)

#define occaGlobalFor0 occaOuterFor0 occaInnerFor0
#define occaGlobalFor1 occaOuterFor1 occaInnerFor1
#define occaGlobalFor2 occaInnerFor2

#define occaGlobalId0  ( occaInnerId0 + occaInnerDim0*occaOuterId0 )
#define occaGlobalId1  ( occaInnerId1 + occaInnerDim1*occaOuterId1 )
#define occaGlobalId2  ( occaInnerId2 )

#define occaBarrier

#define occaShared

#define occaPointer

#define occaVariable &

#define occaRestrict

#define occaVolatile

#define occaAligned __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)))

#define occaConst    const
#define occaConstant

#if 1
#define occaKernelInfoArg const int *occaDims
#define occaFunctionInfoArg const int *occaDims,\
int occaInnerId0,				\
int occaInnerId1, \
int occaInnerId2
#define occaFunctionInfo  occaDims,\
occaInnerId0, \
occaInnerId1, \
occaInnerId2

#define occaKernel extern "C"
#define occaFunction
#else
#define occaKernel(KERNEL, ...) extern "C" void KERNEL(const int *occaDims, __VA_ARGS__)

#define occaFunction(FUNCTION, ...) void FUNCTION(const int *occaDims, int occaInnerId0, int occaInnerId1, int occaInnerId2, __VA_ARGS__)

#define occaFunctionCall(FUNCTION, ...) FUNCTION(occaDims, occaInnerId0, occaInnerId1, occaInnerId2, __VA_ARGS__)
#endif

#define occaLocalMemFence

#define occaGlobalMemFence

#define occaFunctionShared


#define occaDeviceFunction

template <class T> class occaPrivateClass {
private:

public:
  int dim0, dim1, dim2;

  int *id0, *id1, *id2;
#if 0
  T *data;
#else
  // warning hard code (256 max threads)
  T data[512] occaAligned;
#endif

#if 0  
  occaPrivateClass(){
    //    data = NULL;
    dim0 = 0;
    dim1 = 0;
    dim2 = 0;
    id0 = NULL;
    id1 = NULL;
    id2 = NULL;
  }
#endif
  void initialize(int _dim0, int _dim1, int _dim2, int *_id0, int *_id1, int *_id2){
    dim0 = _dim0;
    dim1 = _dim1;
    dim2 = _dim2;
    //    data = new T[dim2*dim1*dim0];
    id0  = _id0;
    id1  = _id1;
    id2  = _id2;
  }

  ~occaPrivateClass(){
    //    if(data)
      //      delete [] data;
  }

  inline int index(){
    int ind = (*id2)*dim0*dim1+(*id1)*dim0 + (*id0);
    return ind;
  }

  inline operator T(){
    return data[index()];
  }

  inline occaPrivateClass<T> & operator= (const T &a){
    data[index()] = a;
    return *this;
  }

  inline occaPrivateClass<T> & operator+= (const T &a){
    data[index()] += a;
    return *this;
  }

  inline occaPrivateClass<T> & operator-= (const T &a){
    data[index()] -= a;
    return *this;
  }

  inline occaPrivateClass<T> & operator/= (const T &a){
    data[index()] /= a;
    return *this;
  }

  inline occaPrivateClass<T> & operator*= (const T &a){
    data[index()] *= a;
    return *this;
  }
};

#if 0
#define occaPrivate(type) occaDims0 = occaDims[0]; \
  occaDims1 = occaDims[1];			       \
  occaDims2 = occaDims[2];			       \
  occaPrivateClass<type>
#endif

#define occaPrivateArray(type, name, sz) \
  occaPrivateClass<type> name[sz];				\
  for(int _n=0;_n<sz;++_n) \
    name[_n].initialize(occaDims[0], occaDims[1], occaDims[2], &occaInnerId0,&occaInnerId1,&occaInnerId2);

#define occaPrivate(type, name) \
  occaPrivateClass<type> name; \
  name.initialize(occaDims[0], occaDims[1], occaDims[2], &occaInnerId0,&occaInnerId1,&occaInnerId2);


#define occaUnroll 
//_Pragma("unroll 16")

#define occaInnerReturn {continue;}

template <class T> T occaAtomicAdd(T* p, T val) { T old = *p; *p += val; return old; };
template <class T> T occaAtomicSub(T* p, T val) { T old = *p; *p -= val; return old; };

#endif
