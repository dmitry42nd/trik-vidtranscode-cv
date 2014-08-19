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

static uint16_t s_hi2ho[IMG_WIDTH_MAX];
static uint8_t  s_metapixFillerShifter[IMG_WIDTH_MAX];

class BitmapBuilder : public CVAlgorithm
{
  private:
    uint64_t m_detectRange;
    uint32_t m_detectExpected;
  
    TrikCvImageDesc m_inImageDesc;
    TrikCvImageDesc m_outImageDesc;

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

      uint16_t* p_hi2ho = s_hi2ho;
      for(uint16_t i = 0; i < m_inImageDesc.m_height; i++) {
        *(p_hi2ho++) = (i / METAPIX_SIZE) * m_outImageDesc.m_width;
      }

      uint8_t* p_metapixFillerShifter = s_metapixFillerShifter;
      for(uint16_t i = 0; i < m_inImageDesc.m_height; i++) {
        *(p_metapixFillerShifter++) = (i % METAPIX_SIZE) * METAPIX_SIZE;
      }

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

      if (detectHueFrom <= detectHueTo) {
        m_detectRange = _itoll((detectValFrom<<16) | (detectSatFrom<<8) | detectHueFrom,
                               (detectValTo  <<16) | (detectSatTo  <<8) | detectHueTo  );
        m_detectExpected = 0x0;
      } else {
        assert(detectHueFrom > 0 && detectHueTo < 255);
        m_detectRange = _itoll((detectValFrom<<16) | (detectSatFrom<<8) | (detectHueTo  +1),
                               (detectValTo  <<16) | (detectSatTo  <<8) | (detectHueFrom-1));
        m_detectExpected = 0x1;
      }

      const uint64_t u64_hsv_range  = m_detectRange;
      const uint32_t u32_hsv_expect = m_detectExpected;

/*
      METAPIX:
      15 14 12 12 11 10 9 8 7 6 5 4 3 2 1 0:
      -------------
      |0 |1 |2 |3 |
      |--|--|--|--|
      |4 |5 |6 |7 |
      |--|--|--|--|
      |8 |9 |10|11|
      |--|--|--|--|
      |12|13|14|15|
      -------------
*/
      const uint64_t* restrict p_inImg = reinterpret_cast<const uint64_t*>(_inImage.m_ptr);
      const uint16_t* restrict p_hi2ho = s_hi2ho;


//just detect and build metapixels:
      uint8_t* restrict p_metapixFillerShifter = s_metapixFillerShifter;
      uint8_t metapixFiller = 0;
      #pragma MUST_ITERATE(4, ,4)
      for (TrikCvImageDimension srcRow = 0; srcRow < m_inImageDesc.m_height; srcRow++) {
        uint16_t* restrict p_outImg = reinterpret_cast<uint16_t*>(_outImage.m_ptr) + *(p_hi2ho++);
        const uint16_t metapixFillerShifter = *(p_metapixFillerShifter++); //(0 4 8 12)...

        #pragma MUST_ITERATE(32, ,32)
        for (TrikCvImageDimension srcCol = 0; srcCol < m_inImageDesc.m_width; srcCol++) {
          bool det = detectHsvPixel(_loll(*(p_inImg++)), u64_hsv_range, u32_hsv_expect);
          *p_outImg += det << (metapixFillerShifter + metapixFiller++);

          if(metapixFiller == METAPIX_SIZE) {
            p_outImg++;
            metapixFiller = 0;
          }
/*
          p_outImg += !(metapixFiller < METAPIX_SIZE);
          metapixFiller %= METAPIX_SIZE;
*/
        }
      }

    }
};


} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_REFERENCE_HPP_
