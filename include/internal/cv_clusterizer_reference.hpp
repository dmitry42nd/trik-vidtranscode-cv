#ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_CV_CLUSTERIZER_REFERENCE_HPP_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_CV_CLUSTERIZER_REFERENCE_HPP_

#ifndef __cplusplus
#error C++-only header
#endif

#include <cassert>
#include <cmath>
#include <vector>

#include "internal/stdcpp.hpp"
#include "trik_vidtranscode_cv.h"


/* **** **** **** **** **** */ namespace trik /* **** **** **** **** **** */ {

/* **** **** **** **** **** */ namespace cv /* **** **** **** **** **** */ {


class Clusterizer : public CVAlgorithm
{
  private:
    TrikCvImageDesc m_inImageDesc;
    TrikCvImageDesc m_outImageDesc;

    std::vector<uint16_t> equalClusters;
    uint16_t m_maxCluster;

    uint16_t min(uint16_t* envPixs)
    {
      uint16_t min = NO_CLUSTER;

      #pragma MUST_ITERATE(4,,4)
        for(int i = 0; i < ENV_PIXS; i++)
          min = envPixs[i] < min ? envPixs[i] : min;

        return min;
    }

    bool isClustersEqual(const uint16_t cluster1, const uint16_t cluster2)
    {
      if (cluster1 == cluster2) 
        return true;
      else
        return (equalClusters[cluster1] == equalClusters[cluster2]);
    }

/*
 ________
|a2|a3|a4|
|a1|p |  |
|  |  |  |
 --------
*/

    void setPixelEnvironment(uint16_t* pixPtr, uint16_t* a, int r, int c)
    {
      const uint32_t width = m_inImageDesc.m_width;

      if(r != 0)
      {
        a[2] = *(pixPtr - width);
        if(c != 0)
          a[1] = *(pixPtr - width - 1);
        if(c != width - 1)
          a[3] = *(pixPtr - width + 1);
      }
      if(c != 0)
        a[0] = *(pixPtr - 1);     
    }


    void setClusterNum(uint16_t* pixPtr, int r, int c)
    {
        uint16_t localMinCluster = NO_CLUSTER;

        uint16_t a[ENV_PIXS];
        memset(a, 0xff, ENV_PIXS*sizeof(uint16_t));
        setPixelEnvironment(pixPtr, a, r, c);

        if((localMinCluster = min(a)) == NO_CLUSTER) {
          *pixPtr = m_maxCluster;
          equalClusters.push_back(m_maxCluster++);
        } else {
          *pixPtr = localMinCluster;

          #pragma MUST_ITERATE(4,,4)
          for(int i = 0; i < ENV_PIXS; i++)
            if((a[i] != NO_CLUSTER))
              if(!isClustersEqual(a[i],localMinCluster))
                equalClusters[a[i]] = equalClusters[localMinCluster];
        }
    }

  public:
    uint16_t getMinEqCluster(uint16_t a)
    {
      return a == NO_CLUSTER ? NO_CLUSTER : equalClusters[a];
    }

    uint16_t getClustersAmount()
    {
      return equalClusters.size();
    }

    uint32_t getObjectsAmount()
    {
      uint32_t n = 0;
      for(int i = 0; i < equalClusters.size(); i++)
        n = n < equalClusters[i] ? equalClusters[i] : n;
      
      return n;
    }

    virtual bool setup(const TrikCvImageDesc& _inImageDesc, const TrikCvImageDesc& _outImageDesc, int8_t* _fastRam, size_t _fastRamSize)
    {
      m_inImageDesc  = _inImageDesc;
      m_outImageDesc = _outImageDesc;

      if (   m_inImageDesc.m_width < 0
          || m_inImageDesc.m_height < 0
          || m_inImageDesc.m_width  % 32 != 0
          || m_inImageDesc.m_height % 4  != 0)
        return false;

      return true;
    }

    virtual bool run(const TrikCvImageBuffer& _inImage, TrikCvImageBuffer& _outImage,
                     const TrikCvAlgInArgs& _inArgs, TrikCvAlgOutArgs& _outArgs)
    {
      m_maxCluster = 0;
      equalClusters.clear();

      const uint16_t* restrict srcImgPtr = reinterpret_cast<uint16_t*>(_inImage.m_ptr);
      uint16_t* restrict dstImgPtr       = reinterpret_cast<uint16_t*>(_outImage.m_ptr);

      for (int srcRow = 0; srcRow < m_inImageDesc.m_height; srcRow++) {
        for (int srcCol = 0; srcCol < m_inImageDesc.m_width; srcCol++) {
          if(pop(*(srcImgPtr++)) > 3) {
            setClusterNum(dstImgPtr, srcRow, srcCol);
          }

          dstImgPtr++; //cause we use addres of dstImgPtr in setClusterNum()
        }
      }
    }
};


} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_REFERENCE_HPP_
