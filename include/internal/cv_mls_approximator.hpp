#ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_CV_MLS_APPROXIMATOR_HPP_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_CV_MLS_APPROXIMATOR_HPP_

#ifndef __cplusplus
#error C++-only header
#endif

#include <stdlib.h>
#include <stdint.h>

#include <cassert>
#include <cmath>
#include <limits>

#include "internal/stdcpp.hpp"


/* **** **** **** **** **** */ namespace trik /* **** **** **** **** **** */ {

/* **** **** **** **** **** */ namespace cv /* **** **** **** **** **** */ {

class MlsApproximator
{
private:
  static const int MAX_N = 8;
  static const int MAX_K = 3;
  int N;
  int K;
  float sums[MAX_K+1][MAX_K+1];
  float a[MAX_K+1];
  float b[MAX_K+1];
  float x[MAX_N];
  float y[MAX_N];

  void init()
  {
    int i=0,j=0, k=0;

    //init square sums matrix
    for(i=0; i<K+1; i++)
    {
      for(j=0; j<K+1; j++)
      {
        sums[i][j] = 0;
        for(k=0; k<N; k++)
        {
          sums[i][j] += pow(x[k], i+j);
        }
      }
    }

    //init free coefficients column
    for(i=0; i<K+1; i++)
    {
      for(k=0; k<N; k++)
      {
        b[i] += pow(x[k], i) * y[k];
      }
    }
  }

/*
  void printresult()
  {
  //print polynom parameters
    int i=0;
    printf("\n");
    for(i=0; i<K+1; i++)
    {
      printf("a[%d] = %f\n", i, a[i]);
    }
  }
*/

  void diagonal()
  {
    int i, j, k;
    float temp=0;
    for(i=0; i<K+1; i++)
    {
      if(sums[i][i]==0)
      {
        for(j=0; j<K+1; j++)
        {
          if(j==i) continue;
          if(sums[j][i] !=0 && sums[i][j]!=0)
          {
            for(k=0; k<K+1; k++)
            {
              temp = sums[j][k];
              sums[j][k] = sums[i][k];
              sums[i][k] = temp;
            }
            temp = b[j];
            b[j] = b[i];
            b[i] = temp;
            break;
          }
        }
      }
    }
  }

public:
  //TODO: initialize arrays
  MlsApproximator(int* _xs, int* _ys, int _N, int _K):
  N(_N),
  K(_K)
  {
    for(int i = 0; i < N; i++)
    {
      y[i] = static_cast<float>(_ys[i]);
      x[i] = static_cast<float>(_xs[i]);
    }
    memset(sums, 0, sizeof(float)*(MAX_K + 1)*(MAX_K + 1));
    memset(a, 0, sizeof(float)*(MAX_K + 1));
    memset(b, 0, sizeof(float)*(MAX_K + 1));
  }

  ~MlsApproximator()
  {}

  int foo(int x)
  {
    int result = 0;

    for(int i = 0; i < K + 1; i++)
    {
      result += a[i]*static_cast<float>(pow(x,i));
    }
    return result;
  }

  int angle()
  {
    return atan(a[1])*180/3.14159; // =(
  }

  void approximate()
  {
    int i=0,j=0, k=0;

    init();
    //check if there are 0 on main diagonal and exchange rows in that case
    diagonal();

    //process rows
    for(k=0; k<K+1; k++)
    {
      for(i=k+1; i<K+1; i++)
      {
        if(sums[k][k]==0)
        {
          //Solution is not exist
          return;
        }
        float M = sums[i][k] / sums[k][k];
        for(j=k; j<K+1; j++)
        {
          sums[i][j] -= M * sums[k][j];
        }
        b[i] -= M*b[k];
      }
    }

    for(i=(K+1)-1; i>=0; i--){
       float s = 0;
       for(j = i; j<K+1; j++){
       s = s + sums[i][j]*a[j];
       }
       a[i] = (b[i] - s) / sums[i][i];
    }
//    printresult();
  }


};

} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_MLS_APPROXIMATOR_HPP_
