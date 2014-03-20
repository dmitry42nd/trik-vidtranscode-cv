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
    TrikCvImageDesc m_inImageDesc;
    TrikCvImageDesc m_outImageDesc;

    std::vector<TrikCvImageDimension> m_srcToDstColConv;
    std::vector<TrikCvImageDimension> m_srcToDstRowConv;
    std::vector<XDAS_UInt16> m_mult255_div;
    std::vector<XDAS_UInt16> m_mult43_div;

    uint16_t max_factor_num; // = 1;
    const uint16_t NO_FACTOR = 0xFF;


    uint16_t pop(uint16_t x)
    {
      x = x - ((x >> 1) & 0x5555);
      x = (x & 0x3333) + ((x >> 2) & 0x3333);
      x = (x + (x >> 4)) & 0x0f0f;
      x = x + (x >> 8);
      x = x + (x >> 16);

      return x & 0x003f;
    }


    uint16_t min(uint16_t** a, int size)
    {
        uint16_t min = NO_FACTOR;

        for(int i = 0; i < size; i++)
            min = *(a[i]) < min? *(a[i]) : min;

    return min;
    }


    void setPixelEnvironment(uint16_t* pixPtr, uint16_t** a1, uint16_t** a2, uint16_t** a3, uint16_t** a4, int r, int c)
    {
        if(r != 0)
        {
            *(a3) = pixPtr - m_inImageDesc.m_width;
            if(c != 0)
                *(a2) = *(a3) - 1;
            if(c != m_inImageDesc.m_width - 1)
                *(a4) = *(a3) + 1;
        }

        if(c != 0)
            *(a1) = pixPtr - 1;
    }


    void setPixelFullEnvironment(uint16_t* pixPtr, uint16_t** a1, uint16_t** a2, uint16_t** a3, uint16_t** a4,
                                                   uint16_t** a5, uint16_t** a6, uint16_t** a7, uint16_t** a8, int r, int c)
    {
        if(r >= 0 && r < m_inImageDesc.m_height && c >= 0 && c < m_inImageDesc.m_width)
        {
            if(r < m_inImageDesc.m_height - 1)
            {
                *a7 = pixPtr + m_inImageDesc.m_width;
                if(c < m_inImageDesc.m_width - 1)
                {
                    *a5 = pixPtr + 1;
                    *a6 = *a5 + m_inImageDesc.m_width;
                }
                if (c > 0)
                {
                    *a1 = pixPtr - 1;
                    *a8 = *a1 + m_inImageDesc.m_width;
                }
            }
            if(r > 0)
            {
                *a3 = pixPtr - m_inImageDesc.m_width;
                if(c < m_inImageDesc.m_width - 1)
                {
                    *a5 = pixPtr + 1;
                    *a4 = *a5 - m_inImageDesc.m_width;
                }
                if (c > 0)
                {
                    *a1 = pixPtr - 1;
                    *a2 = *a1 - m_inImageDesc.m_width;
                }
            }
        }
    }


    void setFactorNum(uint16_t* pixPtr, int r, int c)
    {
        uint16_t local_min_factor_num = NO_FACTOR;

        const uint16_t envPixNum = 4;
        uint16_t* a[envPixNum];
        for (int i =0 ; i < envPixNum; i++)
            a[i] = &(local_min_factor_num);

        setPixelEnvironment(pixPtr, &(a[0]), &(a[1]), &(a[2]), &(a[3]), r, c);

        *pixPtr = max_factor_num;


        if((local_min_factor_num = min(a, envPixNum)) == NO_FACTOR)
        {
            *pixPtr = max_factor_num++;
        }
        else
        {
            *pixPtr = local_min_factor_num;

            if(*(a[0]) != NO_FACTOR && *(a[0]) != local_min_factor_num)
                resetFactorNum((a[0]), local_min_factor_num, r, c-1);

            for(int i = 1; i < 4; i++)
                if(*(a[i]) != NO_FACTOR && *(a[i]) != local_min_factor_num)
                    resetFactorNum((a[i]), local_min_factor_num, r-1, c+i-2);
        }
    }


    void resetFactorNum(uint16_t* pixPtr, int factor_num, int r, int c)
    {
        uint16_t default_factor = NO_FACTOR;

        const uint16_t envPixNum = 8;
        uint16_t* a[envPixNum];
        #pragma MUST_ITERATE(8, ,8)
        for (int i =0 ; i < envPixNum; i++)
            a[i] = &default_factor;

        *pixPtr = factor_num;


        setPixelFullEnvironment(pixPtr, &(a[0]), &(a[1]), &(a[2]), &(a[3]), &(a[4]), &(a[5]), &(a[6]), &(a[7]), r, c);
/*
        if (min(a, envPixNum) != NO_FACTOR)
        {
            if(*(a[0]) != NO_FACTOR && *(a[0]) != factor_num)
                resetFactorNum((a[0]), factor_num, r, c-1);

            for(int i = 1; i < 4; i++)
                if(*(a[i]) != NO_FACTOR && *(a[i]) != factor_num)
                    resetFactorNum((a[i]), factor_num, r-1, c+i-2);

            if(*(a[4]) != NO_FACTOR && *(a[4]) != factor_num)
                resetFactorNum((a[4]), factor_num, r, c+1);

            for(int i = 5; i < 8;i++)
                if(*(a[i]) != NO_FACTOR && *(a[i]) != factor_num)
                    resetFactorNum((a[i]), factor_num, r+1, c+i-6);
        }
*/
    }



  public:
    virtual bool setup(const TrikCvImageDesc& _inImageDesc, const TrikCvImageDesc& _outImageDesc, int8_t* _fastRam, size_t _fastRamSize)
    {
      m_inImageDesc  = _inImageDesc;
      m_outImageDesc = _outImageDesc;

      if (   m_inImageDesc.m_width < 0
          || m_inImageDesc.m_height < 0
          || m_inImageDesc.m_width  % 32 != 0
          || m_inImageDesc.m_height % 4  != 0)
        return false;

      m_srcToDstColConv.resize(m_inImageDesc.m_width);
      for (TrikCvImageDimension srcCol=0; srcCol < m_srcToDstColConv.size(); ++srcCol)
        m_srcToDstColConv[srcCol] = (srcCol*m_outImageDesc.m_width) / m_inImageDesc.m_width;

      m_srcToDstRowConv.resize(m_inImageDesc.m_height);
      for (TrikCvImageDimension srcRow=0; srcRow < m_srcToDstRowConv.size(); ++srcRow)
        m_srcToDstRowConv[srcRow] = (srcRow*m_outImageDesc.m_height) / m_inImageDesc.m_height;

      m_mult255_div.resize(0x100);
      m_mult255_div[0] = 0;
      for (XDAS_UInt16 idx = 1; idx < m_mult255_div.size(); ++idx)
        m_mult255_div[idx] = (static_cast<XDAS_UInt16>(255) * static_cast<XDAS_UInt16>(1u<<8)) / idx;

      m_mult43_div.resize(0x100);
      m_mult43_div[0] = 0;
      for (XDAS_UInt16 idx = 1; idx < m_mult43_div.size(); ++idx)
        m_mult43_div[idx] = (static_cast<XDAS_UInt16>(43) * static_cast<XDAS_UInt16>(1u<<8)) / idx;

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

      max_factor_num = 1;

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
                  setFactorNum(dstImgPtr, dstRow, dstCol);
              }
              srcImgPtr++;
              dstImgPtr++;
          }
      }

#ifdef DEBUG_REPEAT
      } // repeat
#endif
    }
};


} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_REFERENCE_HPP_
