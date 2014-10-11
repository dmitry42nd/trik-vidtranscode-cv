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
    std::vector<uint16_t> directEqualClusters;
    std::vector<uint16_t> clustersMass;

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
      if (cluster1 == cluster2) return true;

      uint16_t tmp = cluster1;
      while(cluster1 != equalClusters[tmp]) {
        tmp = equalClusters[tmp];
        if (tmp == cluster2) return true;
      }

      return false;
    }


    void linkClusters(const uint16_t cluster1, const uint16_t cluster2)
    {
      uint16_t tmp = equalClusters[cluster1];
      equalClusters[cluster1] = equalClusters[cluster2];
      equalClusters[cluster2] = tmp;
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

      a[0] = *(pixPtr - 1);
      a[1] = *(pixPtr - width - 1);
      a[2] = *(pixPtr - width);
      a[3] = *(pixPtr - width + 1);
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
            if((a[i] != NO_CLUSTER) && (a[i] != localMinCluster))
              if(!isClustersEqual(localMinCluster, a[i]))
                linkClusters(localMinCluster, a[i]);
        }

    }


    void setMinEqClusters()
    {
      directEqualClusters.resize(equalClusters.size());

      for(uint16_t c = 1; c < equalClusters.size(); c++) {
        if (c != equalClusters[c]) {
          uint16_t min = NO_CLUSTER;
          uint16_t tmp = c;

          do {
              min = tmp < min ? tmp : min;
              tmp = equalClusters[tmp];
          }
          while(c != tmp);

          directEqualClusters[c] = min;
        }
        else {
          directEqualClusters[c] = c;
        }
      }
    }


  public:
    uint16_t getMinEqCluster(uint16_t a)
    {
      return a == NO_CLUSTER ? NO_CLUSTER : directEqualClusters[a];
    }

    uint16_t getClustersAmount()
    {
      return equalClusters.size();
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
      m_maxCluster = 1;
      equalClusters.resize(1);

      const uint16_t* restrict srcImgPtr = reinterpret_cast<uint16_t*>(_inImage.m_ptr + 1);
      uint16_t* restrict dstImgPtr       = reinterpret_cast<uint16_t*>(_outImage.m_ptr + 1);

      for (int srcRow = 1; srcRow < m_inImageDesc.m_height-1; srcRow++) {
        const uint16_t* restrict srcImgRow = srcImgPtr + srcRow*m_inImageDesc.m_width;
        uint16_t* restrict dstImgRow = dstImgPtr + srcRow*m_outImageDesc.m_width;

        for (int srcCol = 1; srcCol < m_inImageDesc.m_width-1; srcCol++) {
          if(pop(*(srcImgRow++)) > 3) {
            setClusterNum(dstImgRow, srcRow, srcCol);
          }

          dstImgRow++; //cause we use addres of dstImgPtr in setClusterNum()
        }
      }

      setMinEqClusters();
    }
};


} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_REFERENCE_HPP_
