#ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_REFERENCE_HPP_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_REFERENCE_HPP_

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


class BitmapBuilder : public CVAlgorithm
{
  private:
    static const int m_metaPixelSize = 4;

    uint64_t m_detectRange;
    uint32_t m_detectExpected;
  
    TrikCvImageDesc m_inImageDesc;
    TrikCvImageDesc m_outImageDesc;

    std::vector<TrikCvImageDimension> m_srcToDstColConv;
    std::vector<TrikCvImageDimension> m_srcToDstRowConv;
    std::vector<XDAS_UInt16> m_mult255_div;
    std::vector<XDAS_UInt16> m_mult43_div;

    static bool __attribute__((always_inline)) detectHsvPixel(const uint32_t _hsv,
                                                              const uint64_t _hsv_range,
                                                              const uint32_t _hsv_expect)
    {
      const uint32_t u32_hsv_det = _cmpltu4(_hsv, _hill(_hsv_range))
                                 | _cmpgtu4(_hsv, _loll(_hsv_range));

      return (u32_hsv_det == _hsv_expect);
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

      uint32_t detectHueFrom = range<XDAS_Int16>(0, (_inArgs.detectHueFrom * 255) / 359, 255); // scaling 0..359 to 0..255
      uint32_t detectHueTo   = range<XDAS_Int16>(0, (_inArgs.detectHueTo   * 255) / 359, 255); // scaling 0..359 to 0..255
      uint32_t detectSatFrom = range<XDAS_Int16>(0, (_inArgs.detectSatFrom * 255) / 100, 255); // scaling 0..100 to 0..255
      uint32_t detectSatTo   = range<XDAS_Int16>(0, (_inArgs.detectSatTo   * 255) / 100, 255); // scaling 0..100 to 0..255
      uint32_t detectValFrom = range<XDAS_Int16>(0, (_inArgs.detectValFrom * 255) / 100, 255); // scaling 0..100 to 0..255
      uint32_t detectValTo   = range<XDAS_Int16>(0, (_inArgs.detectValTo   * 255) / 100, 255); // scaling 0..100 to 0..255

      if (detectHueFrom <= detectHueTo)
      {
        m_detectRange = _itoll((detectValFrom<<16) | (detectSatFrom<<8) | detectHueFrom,
                               (detectValTo  <<16) | (detectSatTo  <<8) | detectHueTo  );
        m_detectExpected = 0x0;
      }
      else
      {
        assert(detectHueFrom > 0 && detectHueTo < 255);
        m_detectRange = _itoll((detectValFrom<<16) | (detectSatFrom<<8) | (detectHueTo  +1),
                               (detectValTo  <<16) | (detectSatTo  <<8) | (detectHueFrom-1));
        m_detectExpected = 0x1;
      }

      const uint64_t u64_hsv_range  = m_detectRange;
      const uint32_t u32_hsv_expect = m_detectExpected;

#ifdef DEBUG_REPEAT
      for (unsigned repeat = 0; repeat < DEBUG_REPEAT; ++repeat) {
#endif

      for (TrikCvImageDimension srcRow = 0; srcRow < m_inImageDesc.m_height; srcRow++) //r = 0
      {
        const TrikCvImageSize srcRowOffset = srcRow * m_inImageDesc.m_lineLength;
        uint64_t* restrict srcImgPtr = reinterpret_cast<uint64_t*>(_inImage.m_ptr + srcRowOffset);

        const TrikCvImageSize dstRowOffset = (srcRow / m_metaPixelSize) * m_outImageDesc.m_lineLength;
        XDAS_UInt16* restrict dstImgPtr = reinterpret_cast<XDAS_UInt16*>(_outImage.m_ptr + dstRowOffset);

        const int metaPixelRowOffset = (srcRow%m_metaPixelSize)*m_metaPixelSize;

        for (TrikCvImageDimension dstCol = 0; dstCol < m_outImageDesc.m_width; dstCol++)
        {
          int metaPixelPos = metaPixelRowOffset;

          #pragma MUST_ITERATE(4, ,4)
          for (TrikCvImageDimension metaPixelCol = 0; metaPixelCol < m_metaPixelSize; metaPixelCol++)
          {
            if(detectHsvPixel(_loll(*srcImgPtr++), u64_hsv_range, u32_hsv_expect))
              *dstImgPtr += 1u << metaPixelPos;

            metaPixelPos++;
          }
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
