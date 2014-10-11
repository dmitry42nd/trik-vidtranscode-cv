#ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BALL_DETECTOR_SEQPASS_HPP_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BALL_DETECTOR_SEQPASS_HPP_

#ifndef __cplusplus
#error C++-only header
#endif

#include <cassert>
#include <cmath>
#include <c6x.h>
#include <map>
#include <algorithm>

#include "internal/stdcpp.hpp"
#include "trik_vidtranscode_cv.h"
#include "internal/cv_hsv_range_detector.hpp"
#include "internal/cv_bitmap_builder.hpp"
#include "internal/cv_clusterizer.hpp"


/* **** **** **** **** **** */ namespace trik /* **** **** **** **** **** */ {

/* **** **** **** **** **** */ namespace cv /* **** **** **** **** **** */ {

#warning Eliminate global var
static uint64_t s_rgb888hsv[IMG_WIDTH_MAX*IMG_HEIGHT_MAX];
static uint16_t s_bitmap[IMG_WIDTH_MAX*IMG_HEIGHT_MAX];
static uint16_t s_clustermap[IMG_WIDTH_MAX*IMG_HEIGHT_MAX];

static uint32_t s_wi2wo_out[IMG_WIDTH_MAX];
static uint32_t s_hi2ho_out[IMG_HEIGHT_MAX];
static uint32_t s_wi2wo_cstr[IMG_WIDTH_MAX];
static uint32_t s_hi2ho_cstr[IMG_HEIGHT_MAX];

template <>
class BallDetector<TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_YUV422P, TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_RGB565X> : public CVAlgorithm
{
  private:
    static const int m_detectZoneScale = 6;

    uint64_t m_detectRange;
    uint32_t m_detectExpected;

    int32_t  m_targetX;
    int32_t  m_targetY;
    uint32_t m_targetPoints;
//--------------
    uint16_t m_clustersAmount;
    std::vector<int32_t>  X1, X2;
    std::vector<int32_t>  Y1, Y2;
    std::vector<uint32_t> SIZE;
//--------------
    TrikCvImageDesc m_inImageDesc;
    TrikCvImageDesc m_outImageDesc;

    TrikCvImageDesc m_bitmapDesc;
    BitmapBuilder   m_bitmapBuilder;
    TrikCvImageBuffer m_bitmap;

    TrikCvImageDesc m_clustermapDesc;
    Clusterizer     m_clusterizer;
    TrikCvImageBuffer m_clustermap;

    TrikCvImageDesc inRgb888HsvImgDesc;
    TrikCvImageBuffer m_inRgb888HsvImg;

    static uint16_t* restrict s_mult43_div;  // allocated from fast ram
    static uint16_t* restrict s_mult255_div; // allocated from fast ram

    template <typename T1, typename T2>
    struct greater_second {
        typedef std::pair<T1, T2> type;
        bool operator ()(type const& a, type const& b) const {
            return a.second > b.second;
        }
    };

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

      const int32_t dstRow = s_hi2ho_out[srcRow];
      const int32_t dstCol = s_wi2wo_out[srcCol];

      const uint32_t dstOfs = dstRow*m_outImageDesc.m_lineLength + dstCol*sizeof(uint16_t);
      writeOutputPixel(reinterpret_cast<uint16_t*>(_outImage.m_ptr+dstOfs), _rgb888);
    }

    void __attribute__((always_inline)) drawOutputCircle(const int32_t _srcCol,
                                                         const int32_t _srcRow,
                                                         const int32_t _srcRadius,
                                                         const TrikCvImageBuffer& _outImage,
                                                         const uint32_t _rgb888) const
    {
      const int32_t widthBot  = 0;
      const int32_t widthTop  = m_inImageDesc.m_width-1;
      const int32_t heightBot = 0;
      const int32_t heightTop = m_inImageDesc.m_height-1;

      int32_t circleError  = 1-_srcRadius;
      int32_t circleErrorY = 1;
      int32_t circleErrorX = -2*_srcRadius;
      int32_t circleX = _srcRadius;
      int32_t circleY = 0;

      drawOutputPixelBound(_srcCol, _srcRow+_srcRadius, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol, _srcRow-_srcRadius, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol+_srcRadius, _srcRow, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      drawOutputPixelBound(_srcCol-_srcRadius, _srcRow, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);

      while (circleY < circleX)
      {
        if (circleError >= 0)
        {
          circleX      -= 1;
          circleErrorX += 2;
          circleError  += circleErrorX;
        }
        circleY      += 1;
        circleErrorY += 2;
        circleError  += circleErrorY;

        drawOutputPixelBound(_srcCol+circleX, _srcRow+circleY, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_srcCol+circleX, _srcRow-circleY, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_srcCol-circleX, _srcRow+circleY, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_srcCol-circleX, _srcRow-circleY, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_srcCol+circleY, _srcRow+circleX, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_srcCol+circleY, _srcRow-circleX, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_srcCol-circleY, _srcRow+circleX, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_srcCol-circleY, _srcRow-circleX, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      }
    }


    void __attribute__((always_inline)) drawOutputFatRectangle(const int32_t _x1,
                                                            const int32_t _x2,
                                                            const int32_t _y1,
                                                            const int32_t _y2,
                                                            const TrikCvImageBuffer& _outImage,
                                                            const uint32_t _rgb888) const
    {
      const int32_t widthBot  = 0;
      const int32_t widthTop  = m_inImageDesc.m_width-1;
      const int32_t heightBot = 0;
      const int32_t heightTop = m_inImageDesc.m_height-1;

      for (int32_t c = _x1; c < _x2; c++)
      {
        drawOutputPixelBound(c, _y1-1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(c,   _y1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(c, _y1+1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);

        drawOutputPixelBound(c, _y2-1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(c,   _y2, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(c, _y2+1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      }

      for (int32_t r = _y1; r < _y2; r++)
      {
        drawOutputPixelBound(_x1-1, r, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_x1,   r, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_x1+1, r, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);

        drawOutputPixelBound(_x2-1, r, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_x2,   r, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
        drawOutputPixelBound(_x2+1, r, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      }
    }

    void __attribute__((always_inline)) drawOutputRectangle(const int32_t _x1,
                                                            const int32_t _x2,
                                                            const int32_t _y1,
                                                            const int32_t _y2,
                                                            const TrikCvImageBuffer& _outImage,
                                                            const uint32_t _rgb888) const
    {
      const int32_t widthBot  = 0;
      const int32_t widthTop  = m_inImageDesc.m_width-1;
      const int32_t heightBot = 0;
      const int32_t heightTop = m_inImageDesc.m_height-1;

      for (int32_t c = _x1; c < _x2; c++)
      {
        drawOutputPixelBound(c,   _y1, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);

        drawOutputPixelBound(c,   _y2, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      }

      for (int32_t r = _y1; r < _y2; r++)
      {
        drawOutputPixelBound(_x1,   r, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);

        drawOutputPixelBound(_x2,   r, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
      }
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


    static bool __attribute__((always_inline)) detectHsvPixel(const uint32_t _hsv,
                                                              const uint64_t _hsv_range,
                                                              const uint32_t _hsv_expect)
    {
      const uint32_t u32_hsv_det = _cmpltu4(_hsv, _hill(_hsv_range))
                                 | _cmpgtu4(_hsv, _loll(_hsv_range));

      return (u32_hsv_det == _hsv_expect);
    }

    static uint64_t DEBUG_INLINE convert2xYuyvToRgb888(const uint32_t _yuyv)
    {
      const int64_t  s64_yuyv1   = _mpyu4ll(_yuyv,
                                             (static_cast<uint32_t>(static_cast<uint8_t>(409/4))<<24)
                                            |(static_cast<uint32_t>(static_cast<uint8_t>(298/4))<<16)
                                            |(static_cast<uint32_t>(static_cast<uint8_t>(516/4))<< 8)
                                            |(static_cast<uint32_t>(static_cast<uint8_t>(298/4))    ));
      const uint32_t u32_yuyv2   = _dotpus4(_yuyv,
                                             (static_cast<uint32_t>(static_cast<uint8_t>(-208/4))<<24)
                                            |(static_cast<uint32_t>(static_cast<uint8_t>(-100/4))<< 8));
      const uint32_t u32_rgb_h   = _add2(_packh2( 0,         _hill(s64_yuyv1)),
                                         (static_cast<uint32_t>(static_cast<uint16_t>(128/4 + (-128*409-16*298)/4))));
      const uint32_t u32_rgb_l   = _add2(_packlh2(u32_yuyv2, _loll(s64_yuyv1)),
                                          (static_cast<uint32_t>(static_cast<uint16_t>(128/4 + (+128*100+128*208-16*298)/4)<<16))
                                         |(static_cast<uint32_t>(static_cast<uint16_t>(128/4 + (-128*516-16*298)/4))));
      const uint32_t u32_y1y1    = _pack2(_loll(s64_yuyv1), _loll(s64_yuyv1));
      const uint32_t u32_y2y2    = _pack2(_hill(s64_yuyv1), _hill(s64_yuyv1));
      const uint32_t u32_rgb_p1h = _clr(_shr2(_add2(u32_rgb_h, u32_y1y1), 6), 16, 31);
      const uint32_t u32_rgb_p1l =      _shr2(_add2(u32_rgb_l, u32_y1y1), 6);
      const uint32_t u32_rgb_p2h = _clr(_shr2(_add2(u32_rgb_h, u32_y2y2), 6), 16, 31);
      const uint32_t u32_rgb_p2l =      _shr2(_add2(u32_rgb_l, u32_y2y2), 6);
      const uint32_t u32_rgb_p1 = _spacku4(u32_rgb_p1h, u32_rgb_p1l);
      const uint32_t u32_rgb_p2 = _spacku4(u32_rgb_p2h, u32_rgb_p2l);
      return _itoll(u32_rgb_p2, u32_rgb_p1);
    }

    static uint32_t DEBUG_INLINE convertRgb888ToHsv(const uint32_t _rgb888)
    {
      const uint32_t u32_rgb_or16    = _unpkhu4(_rgb888);
      const uint32_t u32_rgb_gb16    = _unpklu4(_rgb888);

      const uint32_t u32_rgb_max2    = _maxu4(_rgb888, _rgb888>>8);
      const uint32_t u32_rgb_max     = _clr(_maxu4(u32_rgb_max2, u32_rgb_max2>>8), 8, 31); // top 3 bytes were non-zeroes!
      const uint32_t u32_rgb_max_max = _pack2(u32_rgb_max, u32_rgb_max);

      const uint32_t u32_hsv_ooo_val_x256   = u32_rgb_max<<8; // get max in 8..15 bits
      const uint32_t u32_rgb_min2    = _minu4(_rgb888, _rgb888>>8);
      const uint32_t u32_rgb_min     = _minu4(u32_rgb_min2, u32_rgb_min2>>8); // top 3 bytes are zeroes
      const uint32_t u32_rgb_delta   = u32_rgb_max-u32_rgb_min;

      /* optimized by table based multiplication with power-2 divisor, simulate 255*(max-min)/max */
      const uint32_t u32_hsv_sat_x256       = s_mult255_div[u32_rgb_max]
                                            * u32_rgb_delta;

      /* optimized by table based multiplication with power-2 divisor, simulate 43*(med-min)/(max-min) */
      const uint32_t u32_hsv_hue_mult43_div = _pack2(s_mult43_div[u32_rgb_delta],
                                                     s_mult43_div[u32_rgb_delta]);
      int32_t s32_hsv_hue_x256;
      const uint32_t u32_rgb_cmp = _cmpeq2(u32_rgb_max_max, u32_rgb_gb16);
      if (u32_rgb_cmp == 0)
          s32_hsv_hue_x256 = static_cast<int32_t>((0x10000*0)/3)
                           + static_cast<int32_t>(_dotpn2(u32_hsv_hue_mult43_div,
                                                          _packhl2(u32_rgb_gb16, u32_rgb_gb16)));
      else if (u32_rgb_cmp == 1)
          s32_hsv_hue_x256 = static_cast<int32_t>((0x10000*2)/3)
                           + static_cast<int32_t>(_dotpn2(u32_hsv_hue_mult43_div,
                                                          _packlh2(u32_rgb_or16, u32_rgb_gb16)));
      else // 2, 3
          s32_hsv_hue_x256 = static_cast<int32_t>((0x10000*1)/3)
                           + static_cast<int32_t>(_dotpn2(u32_hsv_hue_mult43_div,
                                                          _pack2(  u32_rgb_gb16, u32_rgb_or16)));

      const uint32_t u32_hsv_hue_x256      = static_cast<uint32_t>(s32_hsv_hue_x256);
      const uint32_t u32_hsv_sat_hue_x256  = _pack2(u32_hsv_sat_x256, u32_hsv_hue_x256);
      const uint32_t u32_hsv               = _packh4(u32_hsv_ooo_val_x256, u32_hsv_sat_hue_x256);
      return u32_hsv;
    }

    void DEBUG_INLINE convertImageYuyvToHsv(const TrikCvImageBuffer& _inImage)
    {
      const uint32_t srcImageRowEffectiveSize       = m_inImageDesc.m_width;
      const uint32_t srcImageRowEffectiveToFullSize = m_inImageDesc.m_lineLength - srcImageRowEffectiveSize;
      const int8_t* restrict srcImageRowY     = _inImage.m_ptr;
      const int8_t* restrict srcImageRowC     = _inImage.m_ptr + m_inImageDesc.m_lineLength*m_inImageDesc.m_height;
      const int8_t* restrict srcImageToY      = srcImageRowY + m_inImageDesc.m_lineLength*m_inImageDesc.m_height;
      uint64_t* restrict rgb888hsvptr         = s_rgb888hsv;

      assert(m_inImageDesc.m_height % 4 == 0); // verified in setup
#pragma MUST_ITERATE(4, ,4)
      while (srcImageRowY != srcImageToY)
      {
        assert(reinterpret_cast<intptr_t>(srcImageRowY) % 8 == 0); // let's pray...
        assert(reinterpret_cast<intptr_t>(srcImageRowC) % 8 == 0); // let's pray...
        const uint32_t* restrict srcImageColY4 = reinterpret_cast<const uint32_t*>(srcImageRowY);
        const uint32_t* restrict srcImageColC4 = reinterpret_cast<const uint32_t*>(srcImageRowC);
        srcImageRowY += srcImageRowEffectiveSize;
        srcImageRowC += srcImageRowEffectiveSize;

        assert(m_inImageDesc.m_width % 32 == 0); // verified in setup
#pragma MUST_ITERATE(32/4, ,32/4)
        while (reinterpret_cast<const int8_t*>(srcImageColY4) != srcImageRowY)
        {
          assert(reinterpret_cast<const int8_t*>(srcImageColC4) != srcImageRowC);

          const uint32_t yy4x = *srcImageColY4++;
          const uint32_t uv4x = _swap4(*srcImageColC4++);

          const uint32_t yuyv12 = (_unpklu4(yy4x)) | (_unpklu4(uv4x)<<8);
          const uint32_t yuyv34 = (_unpkhu4(yy4x)) | (_unpkhu4(uv4x)<<8);

          const uint64_t rgb12 = convert2xYuyvToRgb888(yuyv12);
          *rgb888hsvptr++ = _itoll(_loll(rgb12), convertRgb888ToHsv(_loll(rgb12)));
          *rgb888hsvptr++ = _itoll(_hill(rgb12), convertRgb888ToHsv(_hill(rgb12)));

          const uint64_t rgb34 = convert2xYuyvToRgb888(yuyv34);
          *rgb888hsvptr++ = _itoll(_loll(rgb34), convertRgb888ToHsv(_loll(rgb34)));
          *rgb888hsvptr++ = _itoll(_hill(rgb34), convertRgb888ToHsv(_hill(rgb34)));
        }

        srcImageRowY += srcImageRowEffectiveToFullSize;
        srcImageRowC += srcImageRowEffectiveToFullSize;
      }
    }

    uint16_t max(std::vector<uint32_t> t)
    {
      uint16_t max = 0;
      uint16_t maxId = 0;

      for(int i = 0; i < t.size(); i++)
      {
        if(t[i] > max)
        {
          max = t[i];
          maxId = i;
        }
      }

      return maxId;
    }


    void DEBUG_INLINE proceedImageHsv(TrikCvImageBuffer& _outImage)
    {
        const uint64_t* restrict rgb888hsvptr = s_rgb888hsv;

        const uint32_t width          = m_inImageDesc.m_width;
        const uint32_t height         = m_inImageDesc.m_height;
        const uint32_t dstLineLength  = m_outImageDesc.m_lineLength;

        uint32_t targetPointsPerRow;
        uint32_t targetPointsCol;

        const uint32_t* restrict p_hi2ho_out = s_hi2ho_out;
        const uint32_t* restrict p_hi2ho_cstr = s_hi2ho_cstr;
        assert(m_outImageDesc.m_height % 4 == 0); // verified in setup
  #pragma MUST_ITERATE(4, ,4)
        for (uint32_t srcRow=0; srcRow < height; srcRow++) {
          const uint32_t dstRow = *(p_hi2ho_out++);
          const uint32_t cstrRow = *(p_hi2ho_cstr++);

          uint16_t* restrict dstImageRow = reinterpret_cast<uint16_t*>(_outImage.m_ptr + dstRow*dstLineLength);
          uint16_t* restrict clustermapRow = reinterpret_cast<uint16_t*>(s_clustermap + cstrRow*m_clustermapDesc.m_width);

          targetPointsPerRow = 0;
          targetPointsCol = 0;

          const uint32_t* restrict p_wi2wo_out = s_wi2wo_out;
          const uint32_t* restrict p_wi2wo_cstr = s_wi2wo_cstr;
          #pragma MUST_ITERATE(32, ,32)
          for (uint32_t srcCol=0; srcCol < width; srcCol++) {
            const uint32_t dstCol    = *(p_wi2wo_out++);
            const uint32_t cstrCol   = *(p_wi2wo_cstr++);
            const uint64_t rgb888hsv = *rgb888hsvptr++;

            uint16_t clusterNum = m_clusterizer.getMinEqCluster(*(clustermapRow + cstrCol));
            const bool det = (clusterNum < 0xFFFF);
            targetPointsPerRow += det;

            if(det)
            {
              X1[clusterNum] = X1[clusterNum] < dstCol ? X1[clusterNum] : dstCol;
              X2[clusterNum] = X2[clusterNum] > dstCol ? X2[clusterNum] : dstCol;
              Y1[clusterNum] = Y1[clusterNum] < dstRow ? Y1[clusterNum] : dstRow;
              Y2[clusterNum] = Y2[clusterNum] > dstRow ? Y2[clusterNum] : dstRow;
            }

            writeOutputPixel(dstImageRow+dstCol, det?0x00ffff:_hill(rgb888hsv));
          }
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

      inRgb888HsvImgDesc.m_width = m_inImageDesc.m_width;
      inRgb888HsvImgDesc.m_height = m_inImageDesc.m_height;
      inRgb888HsvImgDesc.m_lineLength = m_inImageDesc.m_width*sizeof(uint64_t);
      inRgb888HsvImgDesc.m_format = TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_RGB888HSV;
      
      m_bitmapDesc.m_width = m_inImageDesc.m_width/METAPIX_SIZE;
      m_bitmapDesc.m_height = m_inImageDesc.m_height/METAPIX_SIZE;
      m_bitmapDesc.m_lineLength = m_bitmapDesc.m_width*sizeof(uint16_t);
      m_bitmapDesc.m_format = TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_METABITMAP;

      m_clustermapDesc = m_bitmapDesc; //i suppose

      m_bitmapBuilder.setup(inRgb888HsvImgDesc, m_bitmapDesc, _fastRam, _fastRamSize);
      m_clusterizer.setup(m_bitmapDesc, m_clustermapDesc, _fastRam, _fastRamSize);

      m_inRgb888HsvImg.m_ptr = reinterpret_cast<TrikCvImagePtr>(s_rgb888hsv);
      m_inRgb888HsvImg.m_size = IMG_WIDTH_MAX*IMG_HEIGHT_MAX*sizeof(uint64_t);

      m_bitmap.m_ptr = reinterpret_cast<TrikCvImagePtr>(s_bitmap);
      m_bitmap.m_size = IMG_WIDTH_MAX*IMG_HEIGHT_MAX*sizeof(uint16_t);

      m_clustermap.m_ptr = reinterpret_cast<TrikCvImagePtr>(s_clustermap);
      m_clustermap.m_size = IMG_WIDTH_MAX*IMG_HEIGHT_MAX*sizeof(uint16_t);

      if (   m_inImageDesc.m_width < 0
          || m_inImageDesc.m_height < 0
          || m_inImageDesc.m_width  % 32 != 0
          || m_inImageDesc.m_height % 4  != 0)
        return false;

      #define min(x,y) x < y ? x : y;
      const double srcToDstShift = min(static_cast<double>(m_outImageDesc.m_width)/m_inImageDesc.m_width, 
                                 static_cast<double>(m_outImageDesc.m_height)/m_inImageDesc.m_height);

      const uint32_t widthIn  = _inImageDesc.m_width;
      //width step for out image
      uint32_t* restrict p_wi2wo_out = s_wi2wo_out;
      for(int i = 0; i < widthIn; i++) {
          *(p_wi2wo_out++) = i*srcToDstShift;
      }
      //width step for cluster map
      uint32_t* restrict p_wi2wo_cstr = s_wi2wo_cstr;
      for(int i = 0; i < widthIn; i++) {
          *(p_wi2wo_cstr++) = i/METAPIX_SIZE;
      }

      const uint32_t heightIn  = _inImageDesc.m_height;
      //height step for out image
      uint32_t* restrict p_hi2ho_out = s_hi2ho_out;
      for(uint32_t i = 0; i < heightIn; i++) {
          *(p_hi2ho_out++) = i*srcToDstShift;
      }
      //height step for cluster map
      uint32_t* restrict p_hi2ho_cstr = s_hi2ho_cstr;
      for(uint32_t i = 0; i < heightIn; i++) {
          *(p_hi2ho_cstr++) = i/METAPIX_SIZE;
      }

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

      bool autoDetectHsv = static_cast<bool>(_inArgs.autoDetectHsv); // true or false

      memset(s_clustermap, 0xff, m_clustermapDesc.m_width*m_clustermapDesc.m_height*sizeof(uint16_t));
      memset(s_bitmap, 0x00, m_bitmapDesc.m_width*m_bitmapDesc.m_height*sizeof(uint16_t));

#ifdef DEBUG_REPEAT
      for (unsigned repeat = 0; repeat < DEBUG_REPEAT; ++repeat) {
#endif

      if (m_inImageDesc.m_height > 0 && m_inImageDesc.m_width > 0)
      {
        convertImageYuyvToHsv(_inImage);

        if (autoDetectHsv)
        {
          HsvRangeDetector rangeDetector = HsvRangeDetector(m_inImageDesc.m_width, m_inImageDesc.m_height, m_detectZoneScale);
          rangeDetector.detect(_outArgs.detectHue, _outArgs.detectHueTolerance,
                               _outArgs.detectSat, _outArgs.detectSatTolerance,
                               _outArgs.detectVal, _outArgs.detectValTolerance,
                               s_rgb888hsv);
        }

        m_bitmapBuilder.run(m_inRgb888HsvImg, m_bitmap, _inArgs, _outArgs);
        m_clusterizer.run(m_bitmap, m_clustermap, _inArgs, _outArgs);

        m_clustersAmount = m_clusterizer.getClustersAmount();


        X1.resize(m_clustersAmount);
        X2.resize(m_clustersAmount);
        Y1.resize(m_clustersAmount);
        Y2.resize(m_clustersAmount);
        SIZE.resize(m_clustersAmount);

        std::fill(X1.begin(), X1.end(), m_outImageDesc.m_width);
        std::fill(X2.begin(), X2.end(), 0);
        std::fill(Y1.begin(), Y1.end(), m_outImageDesc.m_height);
        std::fill(Y2.begin(), Y2.end(), 0);
        std::fill(SIZE.begin(), SIZE.end(), 0);
        m_targetPoints = 0;

        proceedImageHsv(_outImage);
      }

#ifdef DEBUG_REPEAT
      } // repeat
#endif

      //draw taget pointer
      const int step = m_inImageDesc.m_height/m_detectZoneScale;
      const int hHeight = m_inImageDesc.m_height/2;
      const int hWidth = m_inImageDesc.m_width/2;

      drawRgbTargetCenterLine(hWidth - step, hHeight, _outImage, 0xff00ff);
      drawRgbTargetCenterLine(hWidth + step, hHeight, _outImage, 0xff00ff);
      drawRgbTargetCenterLine(hWidth - 2*step,  hHeight, _outImage, 0xff00ff);
      drawRgbTargetCenterLine(hWidth + 2*step, hHeight, _outImage, 0xff00ff);

      drawRgbTargetHorizontalCenterLine(hWidth, hHeight - step, _outImage, 0xff00ff);
      drawRgbTargetHorizontalCenterLine(hWidth, hHeight + step, _outImage, 0xff00ff);
      drawRgbTargetHorizontalCenterLine(hWidth, hHeight - 2*step, _outImage, 0xff00ff);
      drawRgbTargetHorizontalCenterLine(hWidth, hHeight + 2*step, _outImage, 0xff00ff);


      const uint32_t totalSize = m_outImageDesc.m_height*m_outImageDesc.m_width;

      for(int i = 0; i < m_clustersAmount; i++)
      {
        int32_t a = X2[i] - X1[i];
        int32_t b = Y2[i] - Y1[i];
        if(a > 0 && b > 0)
          SIZE[i] = a*b;
      }

      if (m_targetPoints > 0)
      {
        assert(m_inImageDesc.m_height > 0 && m_inImageDesc.m_width > 0); // more or less safe since no target points would be detected otherwise        
        for(int i = 0; i < OBJECTS; i++)
        {
            int j = max(SIZE);
            if(SIZE[j]>10)
            {
              const int32_t targetX = (X2[j] + X1[j])/2;
              const int32_t targetY = (Y2[j] + Y1[j])/2;

              if(i == 0)
              {
                _outArgs.targetX = ((targetX - static_cast<int32_t>(m_inImageDesc.m_width) /2) * 100*2) / static_cast<int32_t>(m_inImageDesc.m_width);
                _outArgs.targetY = ((targetY - static_cast<int32_t>(m_inImageDesc.m_height)/2) * 100*2) / static_cast<int32_t>(m_inImageDesc.m_height);
                _outArgs.targetSize = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<float>(SIZE[j])))*100*4) / static_cast<uint32_t>(m_inImageDesc.m_width + m_inImageDesc.m_height);
                drawOutputFatRectangle(X1[j], X2[j], Y1[j], Y2[j], _outImage, 0xffff00);
              }
              else
                drawOutputRectangle(X1[j], X2[j], Y1[j], Y2[j], _outImage, 0xffff00);

              SIZE[j] = 0;
            }
    
        }
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

