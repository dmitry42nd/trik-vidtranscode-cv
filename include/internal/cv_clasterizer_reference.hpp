#ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_CV_CLASTERIZER_REFERENCE_HPP_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_CV_CLASTERIZER_REFERENCE_HPP_

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


class Clasterizer : public CVAlgorithm
{
  private:
    const uint16_t NO_CLASTER = 0xFFFF;
    const uint16_t ENV_PIXS_NUM = 4;

    TrikCvImageDesc m_inImageDesc;
    TrikCvImageDesc m_outImageDesc;

    std::vector<uint16_t> equalClasters;
    std::vector<uint16_t> directEqualClasters;
    std::vector<uint16_t> clastersMass;

    uint16_t m_currentMaxClasterNum;


    uint16_t pop(uint16_t x)
    {
      x = x - ((x >> 1) & 0x5555);
      x = (x & 0x3333) + ((x >> 2) & 0x3333);
      x = (x + (x >> 4)) & 0x0f0f;
      x = x + (x >> 8);
      x = x + (x >> 16);

      return x & 0x003f;
    }


    uint16_t min(uint16_t** envPixs)
    {
      uint16_t min = NO_CLASTER;

      #pragma MUST_ITERATE(4,,4)
      for(int i = 0; i < ENV_PIXS_NUM; i++)
          min = *(envPixs[i]) < min ? *(envPixs[i]) : min;

        return min;
    }

    bool isClastersEqual(uint16_t claster1, uint16_t claster2)
    {
      if (claster1 == claster2) return true;

      uint16_t tmp = claster1;
      while(claster1 != equalClasters[tmp])
      {
        tmp = equalClasters[tmp];
        if (tmp == claster2) return true;
      }

      return false;
    }


    void linkClasters(uint16_t claster1, uint16_t claster2)
    {
      uint16_t tmp = equalClasters[claster1];
      equalClasters[claster1] = equalClasters[claster2];
      equalClasters[claster2] = tmp;
    }


    bool justOneClaster(uint16_t** envPixs, uint16_t clasterValue)
    {
      for(int i = 0; i < ENV_PIXS_NUM; i++)
        if (*envPixs[i] != NO_CLASTER && *envPixs[i] != clasterValue)
          return false;

      return true;
    }


    void setPixelEnvironment(uint16_t* pixPtr, uint16_t** a1, uint16_t** a2, uint16_t** a3, uint16_t** a4, int r, int c)
    {
      const uint32_t width = m_inImageDesc.m_width;

      if(r != 0)
      {
        *(a3) = pixPtr - width;
        if(c != 0)
          *(a2) = *(a3) - 1;
        if(c != width - 1)
          *(a4) = *(a3) + 1;
      }

      if(c != 0)
        *(a1) = pixPtr - 1;
    }

    void setClasterNum(uint16_t* pixPtr, int r, int c)
    {
        uint16_t localMinClasterNum = NO_CLASTER;
        uint16_t* a[ENV_PIXS_NUM];
    
        //some kind of init by NO_CLASTER for environment pixels
        #pragma MUST_ITERATE(4,,4)
        for (int i =0 ; i < ENV_PIXS_NUM; i++)
            a[i] = &(localMinClasterNum);

        setPixelEnvironment(pixPtr, &(a[0]), &(a[1]), &(a[2]), &(a[3]), r, c);

        if((localMinClasterNum = min(a)) == NO_CLASTER)
        {
            equalClasters.resize(m_currentMaxClasterNum + 1);
            equalClasters[m_currentMaxClasterNum] = m_currentMaxClasterNum;
            *pixPtr = m_currentMaxClasterNum++;
        }
        else if(justOneClaster(a, localMinClasterNum))
        {
            *pixPtr = localMinClasterNum;
        }
        else
        {
            *pixPtr = localMinClasterNum;

            #pragma MUST_ITERATE(4,,4)
            for(int i = 0; i < ENV_PIXS_NUM; i++)
                if(*(a[i]) != NO_CLASTER && *(a[i]) != localMinClasterNum)
                    if(!isClastersEqual(localMinClasterNum, *a[i]))
                        linkClasters(localMinClasterNum, *a[i]);

        }
    }


    void setMinEqClasters()
    {
        directEqualClasters.resize(equalClasters.size());

        for(uint16_t c = 1; c < equalClasters.size(); c++)
        {
            if (c != equalClasters[c])
            {
                uint16_t min = NO_CLASTER;
                uint16_t tmp = c;

                do
                {
                    min = tmp < min ? tmp : min;
                    tmp = equalClasters[tmp];
                }
                while(c != tmp);

                directEqualClasters[c] = min;
            }
            else
            {
                directEqualClasters[c] = c;
            }
        }
    }


  public:
    uint16_t getMinEqClaster(uint16_t a)
    {
      return a == NO_CLASTER ? NO_CLASTER : directEqualClasters[a];
    }

    uint16_t getClastersAmount()
    {
      return equalClasters.size();
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
      if (m_inImageDesc.m_height * m_inImageDesc.m_lineLength > _inImage.m_size)
        return false;
      if (m_outImageDesc.m_height * m_outImageDesc.m_lineLength > _outImage.m_size)
        return false;
      _outImage.m_size = m_outImageDesc.m_height * m_outImageDesc.m_lineLength;

      m_currentMaxClasterNum = 1;
      equalClasters.resize(1);
      directEqualClasters.resize(1);

#ifdef DEBUG_REPEAT
      for (unsigned repeat = 0; repeat < DEBUG_REPEAT; ++repeat) {
#endif

      for (int dstRow = 0; dstRow < m_inImageDesc.m_height; dstRow++)
      {
          const int dstRowOffset = dstRow*m_inImageDesc.m_lineLength;
          uint16_t* restrict dstImgPtr = reinterpret_cast<uint16_t*>(_outImage.m_ptr + dstRowOffset);
          uint16_t* restrict srcImgPtr = reinterpret_cast<uint16_t*>(_inImage.m_ptr + dstRowOffset);

          for (int dstCol = 0; dstCol < m_inImageDesc.m_width; dstCol++)
          {
              if(pop(*srcImgPtr) > 3)
              {
                 setClasterNum(dstImgPtr, dstRow, dstCol);
              }
              srcImgPtr++;
              dstImgPtr++;
          }
      }

      setMinEqClasters();

#ifdef DEBUG_REPEAT
      } // repeat
#endif
    }
};


} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_REFERENCE_HPP_
