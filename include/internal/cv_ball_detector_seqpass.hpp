  #ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BALL_DETECTOR_SEQPASS_HPP_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BALL_DETECTOR_SEQPASS_HPP_

#ifndef __cplusplus
#error C++-only header
#endif

#include <cassert>
#include <cmath>
#include <c6x.h>
//#include <vlib.h>

extern "C" {
#include <include/IMG_ycbcr422pl_to_rgb565.h>
#include <include/IMG_thr_gt2max_8.h>
//#include <include/IMG_thr_le2min_8.h>
#include <include/IMG_sobel_3x3_8.h>
#include <ti/vlib/src/VLIB_xyGradientsAndMagnitude/VLIB_xyGradientsAndMagnitude.h>
#include <ti/vlib/src/VLIB_harrisScore_7x7/VLIB_harrisScore_7x7.h>
#include <ti/vlib/src/VLIB_nonMaxSuppress_7x7_S16/VLIB_nonMaxSuppress_7x7_S16.h>
}

#include "internal/stdcpp.hpp"
#include "trik_vidtranscode_cv.h"
#include "internal/cv_hsv_range_detector.hpp"


/* **** **** **** **** **** */ namespace trik /* **** **** **** **** **** */ {

/* **** **** **** **** **** */ namespace cv /* **** **** **** **** **** */ {



#warning Eliminate global var
static uint16_t s_y[320*240];
static uint16_t s_y2[320*240];
static uint8_t s_cb[320*240];
static uint8_t s_cr[320*240];

static int16_t s_xGrad[320*240+1];
static int16_t s_yGrad[320*240+1];
static int16_t s_gradMag[320*240+1];
static uint16_t s_harrisScore[320*240];

static uint8_t s_buffer[20]; //200

static const short s_coeff[5] = { 0x2000, 0x2BDD, -0x0AC5, -0x1658, 0x3770 };


template <>
class BallDetector<TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_YUV422P, TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_RGB565X> : public CVAlgorithm
{
  private:
    static const int m_detectZoneScale = 6;
/*
  m_detectZoneScale
  The perpose of this variable is to set borders of auto color detector special zones.
  Example:
    Let's m_detectZoneScale is 6. 
    Then "step" is m_width/6
    It means that inside zone of color detector looks like square wiht bounds at (x_center +- step) and (y_center +- step).
    Middle (neutral) zone has (x_center +- 2*step) and (y_center +- 2*step) bounds.

    ----------------
    |   neutral    |
    |   --------   |
    |   |inside|   |
    |   | zone |   |
    |   |      |   |
    |   --------   |
    |    zone      |
    ----------------
*/

    uint64_t m_detectRange;
    uint32_t m_detectExpected;
    double m_srcToDstShift;

    int32_t  m_targetX;
    int32_t  m_targetY;
    uint32_t m_targetPoints;

    TrikCvImageDesc m_inImageDesc;
    TrikCvImageDesc m_outImageDesc;

    static uint16_t* restrict s_mult43_div;  // allocated from fast ram
    static uint16_t* restrict s_mult255_div; // allocated from fast ram

    static void __attribute__((always_inline)) writeOutputPixel(uint16_t* restrict _rgb565ptr,
                                                                const uint32_t _rgb888)
    {
      *_rgb565ptr = ((_rgb888>>19)&0x001f) | ((_rgb888>>5)&0x07e0) | ((_rgb888<<8)&0xf800);
    }

    void __attribute__((always_inline)) drawOutputPixelBound(const int32_t _srcCol,
                                                             const int32_t _srcRow,
                                                             const int32_t _srcColBot,
                                                             const int32_t _srcColTop,
                                                             const int32_t _srcRowBot,
                                                             const int32_t _srcRowTop,
                                                             const TrikCvImageBuffer& _outImage,
                                                             const uint32_t _rgb888) const
    {
      const int32_t srcCol = range<int32_t>(_srcColBot, _srcCol, _srcColTop);
      const int32_t srcRow = range<int32_t>(_srcRowBot, _srcRow, _srcRowTop);
      const double srcToDstShift = m_srcToDstShift;
    
      const int32_t dstRow = srcRow * srcToDstShift;
      const int32_t dstCol = srcCol * srcToDstShift;

      const uint32_t dstOfs = dstRow*m_outImageDesc.m_lineLength + dstCol*sizeof(uint16_t);
      writeOutputPixel(reinterpret_cast<uint16_t*>(_outImage.m_ptr+dstOfs), _rgb888);
    }

    void __attribute__((always_inline)) drawRgbTargetCenterLine(const int32_t _srcCol, 
                                                                const int32_t _srcRow,
                                                                const TrikCvImageBuffer& _outImage,
                                                                const uint32_t _rgb888)
    {
      const int32_t widthBot  = 0;
      const int32_t widthTop  = m_inImageDesc.m_width-1;
      const int32_t heightBot = 0;
      const int32_t heightTop = m_inImageDesc.m_height-1;

      for (int adj = 0; adj < 100; ++adj)
      {
        drawOutputPixelBound(_srcCol  , _srcRow-adj, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_srcCol  , _srcRow+adj, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      }
    }

    void __attribute__((always_inline)) drawCornerHighlight(const int32_t _srcCol, 
                                                                const int32_t _srcRow,
                                                                const TrikCvImageBuffer& _outImage,
                                                                const uint32_t _rgb888)
    {
      const int32_t widthBot  = 0;
      const int32_t widthTop  = m_inImageDesc.m_width-1;
      const int32_t heightBot = 0;
      const int32_t heightTop = m_inImageDesc.m_height-1;

      drawOutputPixelBound(_srcCol-1, _srcRow-1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol-1, _srcRow, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol-1, _srcRow+1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol, _srcRow-1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol, _srcRow, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol, _srcRow+1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol+1, _srcRow-1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol+1, _srcRow, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol+1, _srcRow+1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
    }

    void __attribute__((always_inline)) drawRgbTargetHorizontalCenterLine(const int32_t _srcCol, 
                                                                const int32_t _srcRow,
                                                                const TrikCvImageBuffer& _outImage,
                                                                const uint32_t _rgb888)
    {
      const int32_t widthBot  = 0;
      const int32_t widthTop  = m_inImageDesc.m_width-1;
      const int32_t heightBot = 0;
      const int32_t heightTop = m_inImageDesc.m_height-1;

      for (int adj = 0; adj < 100; ++adj) 
      {
        drawOutputPixelBound(_srcCol-adj  , _srcRow, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_srcCol+adj  , _srcRow, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      }
    }


    void DEBUG_INLINE convertImageYuyvToRgb(const TrikCvImageBuffer& _inImage, TrikCvImageBuffer& _outImage)
    {
      const uint32_t width          = m_inImageDesc.m_width;
      const uint32_t height         = m_inImageDesc.m_height;
      const uint32_t imgSize        = width*height;
      const double srcToDstShift    = m_srcToDstShift;

      uint16_t targetPointsPerRow;
      uint16_t targetPointsCol;
//separate Cb Cr
      uint8_t* restrict cb   = reinterpret_cast<uint8_t*>(s_cb);
      uint8_t* restrict cr   = reinterpret_cast<uint8_t*>(s_cr);
      const uint16_t* restrict CbCr = reinterpret_cast<const uint16_t*>(_inImage.m_ptr +
                                                                    m_inImageDesc.m_lineLength*m_inImageDesc.m_height);
      #pragma MUST_ITERATE(8, ,8)
      for(int i = 0; i < imgSize; i++) {
          *(cb++) = static_cast<uint8_t>(*CbCr);
          *(cr++) = static_cast<uint8_t>((*CbCr) >> 8);
          CbCr++;
      }


//Sobel edge detection
      const unsigned char* restrict y_in_sobel  = reinterpret_cast<const unsigned char*>(_inImage.m_ptr);
      unsigned char* restrict sobel_out = reinterpret_cast<unsigned char*>(s_y);
      IMG_sobel_3x3_8(y_in_sobel, sobel_out, width, height);

      IMG_thr_gt2max_8(reinterpret_cast<const unsigned char*>(s_y), 
                       reinterpret_cast<unsigned char*>(s_y),
                       width, height, 50);

      const uint8_t* restrict y = reinterpret_cast<unsigned char*>(s_y);
      uint8_t* restrict y2 = reinterpret_cast<unsigned char*>(s_y2);
      
      assert(m_inImageDesc.m_height % 4 == 0); // verified in setup
      #pragma MUST_ITERATE(4, ,4)
      for(int r = 0; r < height; r++) {
//        const uint32_t dR = r * srcToDstShift;
//        int8_t* restrict dIR = reinterpret_cast<uint8_t*>(y2 + dR*dstLineLength);

        targetPointsPerRow = 0;
        targetPointsCol = 0;

        assert(m_inImageDesc.m_width % 32 == 0); // verified in setup
#pragma MUST_ITERATE(32, ,32)
        for(int c = 0; c < width; c++) {
//          const uint32_t dC = c * srcToDstShift;
          if(c > 10 && c < width - 10) {
            const bool det = (*y == 0xFF);
            targetPointsPerRow += det;
            targetPointsCol += det?c:0;
          }
          y++;
        }
        m_targetX      += targetPointsCol;
        m_targetPoints += targetPointsPerRow;
      }

/*
//Harris corner detector
      VLIB_xyGradientsAndMagnitude(reinterpret_cast<const uint8_t*>(_inImage.m_ptr), 
                                   reinterpret_cast<int16_t*>(s_xGrad), 
                                   reinterpret_cast<int16_t*>(s_yGrad),
                                   reinterpret_cast<int16_t*>(s_gradMag), width, height);

      const int16_t* restrict xGrad_h = reinterpret_cast<const int16_t*>(s_xGrad);
      const int16_t* restrict yGrad_h = reinterpret_cast<const int16_t*>(s_yGrad);
      int16_t* restrict out_m          = reinterpret_cast<int16_t*>(s_harrisScore);
      uint8_t* restrict buffer         = reinterpret_cast<uint8_t*>(s_buffer); //what for?
      VLIB_harrisScore_7x7(reinterpret_cast<const int16_t*>(s_xGrad),
                           reinterpret_cast<const int16_t*>(s_yGrad),
                           width, height,
                           reinterpret_cast<int16_t*>(s_harrisScore),
                           2500, 
                           reinterpret_cast<uint8_t*>(s_buffer));

      VLIB_nonMaxSuppress_7x7_S16(reinterpret_cast<const int16_t*>(s_harrisScore), 
                                  width, height, 17000, 
                                  reinterpret_cast<uint8_t*>(s_y2));
*/

//in_img to rgb565 & out
      const short* restrict coeff = s_coeff;
      const unsigned char* restrict res_in = reinterpret_cast<const unsigned char*>(s_y);
      const unsigned char* restrict cb_in  = reinterpret_cast<const unsigned char*>(s_cb);
      const unsigned char* restrict cr_in  = reinterpret_cast<const unsigned char*>(s_cr);
      unsigned short* rgb565_out           = reinterpret_cast<unsigned short*>(_outImage.m_ptr);
      IMG_ycbcr422pl_to_rgb565(coeff, res_in, cb_in, cr_in, rgb565_out, width*height);

/*
//    proceedImageHsv(_inImage, _outImage);

//highlight corners
      const uint8_t* restrict corners  = reinterpret_cast<uint8_t*>(s_y2);
      #pragma MUST_ITERATE(8, ,8)
      for(int r = 0; r < 240; r++) {
        #pragma MUST_ITERATE(8, ,8)
        for(int c = 0; c < 320; c++) {
          if(c > 10 && c < 310 && r > 5 && r < 235)
            if (*corners != 0)
              drawCornerHighlight(c, r, _outImage, 0xff0000);
          corners++;
        } 
      }
*/

    }


    static bool __attribute__((always_inline)) detectHsvPixel(const uint32_t _hsv,
                                                              const uint64_t _hsv_range,
                                                              const uint32_t _hsv_expect)
    {
      const uint32_t u32_hsv_det = _cmpltu4(_hsv, _hill(_hsv_range))
                                 | _cmpgtu4(_hsv, _loll(_hsv_range));

      return (u32_hsv_det == _hsv_expect);
    }


    void DEBUG_INLINE proceedImageHsv(const TrikCvImageBuffer& _inImage, TrikCvImageBuffer& _outImage)
    {
      const uint8_t* restrict rgb888hsvptr = reinterpret_cast<const uint8_t*>(_inImage.m_ptr);
      const uint32_t width          = m_inImageDesc.m_width;
      const uint32_t height         = m_inImageDesc.m_height;
      const uint32_t dstLineLength  = m_outImageDesc.m_lineLength;
      const /*uint32_t*/ double srcToDstShift  = m_srcToDstShift;
      const uint64_t u64_hsv_range  = m_detectRange;
      const uint32_t u32_hsv_expect = m_detectExpected;
      uint32_t targetPointsPerRow;
      uint32_t targetPointsCol;

      assert(m_inImageDesc.m_height % 4 == 0); // verified in setup
#pragma MUST_ITERATE(4, ,4)
      for (uint32_t srcRow=0; srcRow < height; ++srcRow)
      {
        const uint32_t dstRow = srcRow * srcToDstShift;
        uint16_t* restrict dstImageRow = reinterpret_cast<uint16_t*>(_outImage.m_ptr + dstRow*dstLineLength);

        targetPointsPerRow = 0;
        targetPointsCol = 0;
        assert(m_inImageDesc.m_width % 32 == 0); // verified in setup
#pragma MUST_ITERATE(32, ,32)
        for (uint32_t srcCol=0; srcCol < width; ++srcCol)
        {
          const uint32_t dstCol    = srcCol * srcToDstShift;
          const uint32_t rgb888hsv = static_cast<const uint32_t>(*rgb888hsvptr++) << 16;

          const bool det = detectHsvPixel(rgb888hsv, u64_hsv_range, u32_hsv_expect);
          targetPointsPerRow += det;
          targetPointsCol += det?srcCol:0;
          if(det)
            writeOutputPixel(dstImageRow+dstCol, 0x00ffff);
        } 
        m_targetX      += targetPointsCol;
        m_targetY      += srcRow*targetPointsPerRow;
        m_targetPoints += targetPointsPerRow;
      }
    }

  public:
    virtual bool setup(const TrikCvImageDesc& _inImageDesc,
                       const TrikCvImageDesc& _outImageDesc,
                       int8_t* _fastRam, size_t _fastRamSize)
    {
      m_inImageDesc  = _inImageDesc;
      m_outImageDesc = _outImageDesc;

      if (   m_inImageDesc.m_width < 0
          || m_inImageDesc.m_height < 0
          || m_inImageDesc.m_width  % 32 != 0
          || m_inImageDesc.m_height % 4  != 0)
        return false;

      #define min(x,y) x < y ? x : y;
      m_srcToDstShift = min(static_cast<double>(m_outImageDesc.m_width)/m_inImageDesc.m_width, 
                            static_cast<double>(m_outImageDesc.m_height)/m_inImageDesc.m_height);

      /* Static member initialization on first instance creation */
      if (s_mult43_div == NULL || s_mult255_div == NULL)
      {
        if (_fastRamSize < (1u<<8)*sizeof(*s_mult43_div) + (1u<<8)*sizeof(*s_mult255_div))
          return false;

        s_mult43_div  = reinterpret_cast<typeof(s_mult43_div)>(_fastRam);
        _fastRam += (1u<<8)*sizeof(*s_mult43_div);
        s_mult255_div = reinterpret_cast<typeof(s_mult255_div)>(_fastRam);
        _fastRam += (1u<<8)*sizeof(*s_mult255_div);

        s_mult43_div[0] = 0;
        s_mult255_div[0] = 0;
        for (uint32_t idx = 1; idx < (1u<<8); ++idx)
        {
          s_mult43_div[ idx] = (43u  * (1u<<8)) / idx;
          s_mult255_div[idx] = (255u * (1u<<8)) / idx;
        }
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

      m_targetX = 0;
      m_targetY = 0;
      m_targetPoints = 0;

      uint32_t detectHueFrom = 0;
      uint32_t detectHueTo   = 255;
      uint32_t detectSatFrom = 0;
      uint32_t detectSatTo   = 255;
      uint32_t detectValFrom = range<int32_t>(0, (static_cast<int32_t>(_inArgs.detectValFrom) * 255) / 100, 255); // scaling 0..100 to 0..255
      uint32_t detectValTo   = range<int32_t>(0, (static_cast<int32_t>(_inArgs.detectValTo  ) * 255) / 100, 255); // scaling 0..100 to 0..255
      bool     autoDetectHsv = static_cast<bool>(_inArgs.autoDetectHsv); // true or false

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

#ifdef DEBUG_REPEAT
      for (unsigned repeat = 0; repeat < DEBUG_REPEAT; ++repeat) {
#endif
        if (m_inImageDesc.m_height > 0 && m_inImageDesc.m_width > 0)
        {
          convertImageYuyvToRgb(_inImage, _outImage);
        }
#ifdef DEBUG_REPEAT
      } // repeat
#endif

      XDAS_Int32 drawY = m_inImageDesc.m_height/2;

      if (m_targetPoints > 0)
      {
        const int32_t targetX = m_targetX/m_targetPoints;
        const int32_t targetY = m_targetY/m_targetPoints;

        assert(m_inImageDesc.m_height > 0 && m_inImageDesc.m_width > 0); // more or less safe since no target points would be detected otherwise
        const uint32_t targetRadius = std::ceil(std::sqrt(static_cast<float>(m_targetPoints) / 3.1415927f));

        drawRgbTargetCenterLine(targetX, drawY, _outImage, 0xff0000);  

        _outArgs.targetX = ((targetX - static_cast<int32_t>(m_inImageDesc.m_width) /2) * 100*2) / static_cast<int32_t>(m_inImageDesc.m_width);
        _outArgs.targetY = ((targetY - static_cast<int32_t>(m_inImageDesc.m_height)/2) * 100*2) / static_cast<int32_t>(m_inImageDesc.m_height);
        _outArgs.targetSize = static_cast<uint32_t>(targetRadius*100*4) / static_cast<uint32_t>(m_inImageDesc.m_width + m_inImageDesc.m_height);

      }
      else
      {
        _outArgs.targetX = 0;
        _outArgs.targetY = 0;
        _outArgs.targetSize = 0;
      }

      return true;
    }
};

uint16_t* restrict BallDetector<TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_YUV422P,
                                TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_RGB565X>::s_mult43_div = NULL;
uint16_t* restrict BallDetector<TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_YUV422P,
                                TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_RGB565X>::s_mult255_div = NULL;


} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BALL_DETECTOR_SEQPASS_HPP_

