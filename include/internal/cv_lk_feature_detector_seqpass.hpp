  #ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BALL_DETECTOR_SEQPASS_HPP_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BALL_DETECTOR_SEQPASS_HPP_

#ifndef __cplusplus
#error C++-only header
#endif

#include <cassert>
#include <cmath>
#include <c6x.h>
#include <string.h>
#include <stdio.h>

extern "C" {
#include <include/IMG_ycbcr422pl_to_rgb565.h>

#include <ti/vlib/vlib.h>
}

#include "internal/stdcpp.hpp"
#include "trik_vidtranscode_cv.h"


/* **** **** **** **** **** */ namespace trik /* **** **** **** **** **** */ {

/* **** **** **** **** **** */ namespace cv /* **** **** **** **** **** */ {

static uint16_t s_rgb[320*240];

static uint32_t s_wi2wo[640];
static uint32_t s_hi2ho[480];

static const short s_coeff[5] = { 0x2000, 0x2BDD, -0x0AC5, -0x1658, 0x3770 };

template <>
class LKFeatureDetector<TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_YUV422P, TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_RGB565X> : public CVAlgorithm
{
  private:
    TrikCvImageDesc m_inImageDesc;
    TrikCvImageDesc m_outImageDesc;

    uint8_t m_scaleFactor;

    uint8_t *m_yIn;
    uint8_t *m_cbIn;
    uint8_t *m_crIn;

    uint8_t *m_yOut;
    uint8_t *m_cbOut;
    uint8_t *m_crOut;

    uint16_t *m_outError;

    int16_t *m_gX;
    int16_t *m_gY;
    int16_t *m_mag;

    int16_t *m_cornersScore;
    uint8_t *m_cornersMap;
    uint8_t *m_buffer;

    uint8_t *m_oldPyrBuf;
    uint8_t *m_newPyrBuf;

    uint8_t *m_oldPyr[3];
    uint8_t *m_newPyr[3];

    int32_t m_totalFeaturesNum;
    int32_t m_currentFeaturesNum;

    uint16_t *m_X;
    uint16_t *m_Y;

    uint16_t *m_newX;
    uint16_t *m_newY;

    uint16_t *m_pyramidX;
    uint16_t *m_pyramidY;
    
    uint8_t *m_scratch;


    uint16_t *outTemp;
    int16_t  *pixIndex;
    uint16_t *internalBuf;
    int32_t  *ind;
    int32_t   good_points_number;


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

      const int32_t dstRow = srcRow;//s_hi2ho[srcRow];
      const int32_t dstCol = srcCol;//s_wi2wo[srcCol];

      const uint32_t dstOfs = dstRow*m_outImageDesc.m_lineLength + dstCol*sizeof(uint16_t);
      writeOutputPixel(reinterpret_cast<uint16_t*>(_outImage.m_ptr+dstOfs), _rgb888);
    }

    void __attribute__((always_inline)) drawDot(const int32_t _srcCol, 
                                                const int32_t _srcRow,
                                                const TrikCvImageBuffer& _outImage,
                                                const uint32_t _rgb888)
    {
      const int32_t widthBot  = 0;
      const int32_t widthTop  = m_inImageDesc.m_width-1;
      const int32_t heightBot = 0;
      const int32_t heightTop = m_inImageDesc.m_height-1;

      drawOutputPixelBound(_srcCol, _srcRow, widthBot, widthTop, heightBot, heightTop, _outImage, _rgb888);
    }

    void drawLine(const int32_t x0, const int32_t y0, 
                  const int32_t x1, const int32_t y1,
                  const TrikCvImageBuffer& _outImage,
                  const uint32_t _rgb888)
    {
      int dx = x1 - x0;
      int dy = y1 - y0;
      int xmin = x0 < x1 ? x0 : x1;
      int xmax = x0 > x1 ? x0 : x1;
      for (int x = xmin; x < xmax; x++) {
        int y = y0 + dy * (x - x0) / dx;
        drawDot(x, y, _outImage, _rgb888);
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


    void YcbcrSeparation(const TrikCvImageBuffer& _inImage) {
      const uint32_t width  = m_inImageDesc.m_width;
      const uint32_t height = m_inImageDesc.m_height;
      
      const uint8_t* restrict CbCr = reinterpret_cast<const uint8_t*>(_inImage.m_ptr + 
                                                                       m_inImageDesc.m_lineLength*height);
      VLIB_convertUYVYsemipl_to_YUVpl(CbCr, width, width, height, m_cbIn, m_crIn);

      m_yIn = reinterpret_cast<uint8_t *>(_inImage.m_ptr);
    }   
    
    
    void detectFeaturesToTrack(const TrikCvImageBuffer& _inImage)
    {
      const uint32_t width          = m_inImageDesc.m_width;
      const uint32_t height         = m_inImageDesc.m_height;

      //tracking Harris Corners
      VLIB_xyGradients(reinterpret_cast<const uint8_t*>(_inImage.m_ptr), m_gX + width + 1, m_gY + width + 1, width, height);
      VLIB_harrisScore_7x7(m_gX, m_gY, width, height, m_cornersScore, 3000, m_buffer);
/*
      VLIB_goodFeaturestoTrack(reinterpret_cast<const uint16_t *>(m_cornersScore),
                               m_cornersMap,
                               width, height,
                               8000, 7, 80, m_totalFeaturesNum, 10,
                               outTemp,  &good_points_number,
                               pixIndex, internalBuf, ind);
*/
      VLIB_nonMaxSuppress_7x7_S16(m_cornersScore, width, height, 8000, m_cornersMap);
    }

    void setFeaturesToTrack() {
      const uint32_t width  = m_inImageDesc.m_width;
      const uint32_t height = m_inImageDesc.m_height;

      m_currentFeaturesNum = 0;
      memset(m_X, 0, m_totalFeaturesNum*sizeof(int16_t));
      memset(m_Y, 0, m_totalFeaturesNum*sizeof(int16_t));
      
      const uint8_t* restrict corners  = reinterpret_cast<uint8_t*>(m_cornersMap);
      for(int r = 0; r < height; r++) {
        for(int c = 0; c < width; c++) {
          if(c > 10 && c < 310 && r > 5 && r < 235)
            if (*corners != 0) {
              if (m_currentFeaturesNum < m_totalFeaturesNum) {
                m_X[m_currentFeaturesNum] = c << 4;
                m_Y[m_currentFeaturesNum] = r << 4;
                m_currentFeaturesNum++;
              }
            }
          corners++;
        }
      }
    }

    void doLKStuff(const TrikCvImageBuffer& _inImage, TrikCvImageBuffer& _outImage) {
      const uint32_t width  = m_inImageDesc.m_width;
      const uint32_t height = m_inImageDesc.m_height;
      
#if 1
      VLIB_imagePyramid8(reinterpret_cast<const uint8_t*>(_inImage.m_ptr), width, height, m_newPyrBuf);

      for(int i = 0; i < m_totalFeaturesNum; i++) {
        m_newX[i] = m_X[i] >> 3;
        m_newY[i] = m_Y[i] >> 3;
      }

      for (int i = 3; i > 0; i--) {
        for (int j = 0; j < m_totalFeaturesNum; j++) {
          m_pyramidX[j] = m_X[j] >> i;
          m_pyramidY[j] = m_Y[j] >> i;
        }

        VLIB_xyGradients(m_oldPyr[i], m_gX + (width >> i) + 1, m_gY + (width >> i) + 1, width >> i, height >> i);
        VLIB_trackFeaturesLucasKanade_7x7(m_oldPyr[i], m_newPyr[i],
                                          m_gX, m_gY,
                                          width >> i, height >> i,
                                          m_totalFeaturesNum,
                                          m_pyramidX, m_pyramidY,
                                          m_newX, m_newY,
                                          m_outError, 10, 0, m_scratch);

        // m_newX, m_newY refined at level i are scaled to become estimates for next iteration
        for (int j = 0; j < m_totalFeaturesNum; j++) {
          m_newX[j] = m_newX[j] << 1;
          m_newY[j] = m_newY[j] << 1;
        }
      }

      VLIB_xyGradients(m_oldPyr[0], m_gX + width + 1, m_gY + width + 1, width, height);
      VLIB_trackFeaturesLucasKanade_7x7(m_oldPyr[0], reinterpret_cast<const uint8_t*>(_inImage.m_ptr), 
                                        m_gX, m_gY, 
                                        width, height,
                                        m_totalFeaturesNum,
                                        m_X, m_Y, 
                                        m_newX, m_newY,
                                        m_outError, 10, 0, m_scratch);

      memcpy(m_oldPyr[0], _inImage.m_ptr, width * height);
      memcpy(m_oldPyrBuf, m_newPyrBuf, width * height * 21 / 64);
#endif

//postprocessing
//scale that muttersaftsack down!
      uint8_t* y_out;
      uint8_t* cb_out;
      uint8_t* cr_out;

      if(m_scaleFactor > 1) {
        const uint8_t* restrict scale_y_in  = reinterpret_cast<const uint8_t*>(m_yIn);
        uint8_t* restrict scale_y_out = reinterpret_cast<uint8_t*>(m_yOut);
        VLIB_image_rescale(scale_y_in, scale_y_out, (1 << 13), width, height, 3);

        const uint8_t* restrict scale_cb_in  = reinterpret_cast<const uint8_t*>(m_cbIn);
        uint8_t* restrict scale_cb_out = reinterpret_cast<uint8_t*>(m_cbOut);
        VLIB_image_rescale(scale_cb_in, scale_cb_out, (1 << 13), width, height, 3);
        
        const uint8_t* restrict scale_cr_in  = reinterpret_cast<const uint8_t*>(m_crIn);
        uint8_t* restrict scale_cr_out = reinterpret_cast<uint8_t*>(m_crOut);
        VLIB_image_rescale(scale_cr_in, scale_cr_out, (1 << 13), width, height, 3);

        y_out   = m_yOut;
        cb_out  = m_cbOut;
        cr_out  = m_crOut;
      } else {
        y_out   = m_yIn;
        cb_out  = m_cbIn;
        cr_out  = m_crIn;
      }

//yuv422pl to rgb565 to outImage
      const short* restrict coeff = s_coeff;
      unsigned short* rgb565_out  = reinterpret_cast<unsigned short*>(_outImage.m_ptr);
      IMG_ycbcr422pl_to_rgb565(coeff, reinterpret_cast<const unsigned char*>(y_out), 
                                      reinterpret_cast<const unsigned char*>(cb_out), 
                                      reinterpret_cast<const unsigned char*>(cr_out), rgb565_out, width*height);
/*
      const uint8_t* restrict corners  = reinterpret_cast<uint8_t*>(m_cornersMap);
      for(int r = 0; r < height; r++) {
        for(int c = 0; c < width; c++) {
          if(c > 10 && c < width - 10 && r > 5 && r < height - 10)
            if (*corners != 0) {
              drawCornerHighlight(c, r, _outImage, 0xff0000);
            }
          corners++;
        }
      }
*/
      for(int i = 0; i < m_totalFeaturesNum; i++) {
//        if(m_outError[i] < 40) {
          drawLine(m_X[i] >> 4, m_Y[i] >> 4, m_newX[i] >> 4, m_newY[i] >> 4, _outImage, 0xffffff);
          drawCornerHighlight(m_newX[i] >> 4, m_newY[i] >> 4, _outImage, 0x00ff00);
          m_X[i] = m_newX[i];
          m_Y[i] = m_newY[i];
          /*
        } else {
          m_X[i] = -1;
          m_Y[i] = -1;
          m_currentFeaturesNum--;
        }
        */
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

      int width  = _inImageDesc.m_width;
      int height = _inImageDesc.m_height;

      #define min(x,y) x < y ? x : y;
      m_scaleFactor = min(static_cast<double>(m_outImageDesc.m_width)/width, 
                          static_cast<double>(m_outImageDesc.m_height)/height);

//malloc zone
//      m_yIn  = (uint8_t *)memalign(64, m_inImageDesc.m_width*m_inImageDesc.m_height*sizeof(uint8_t));
      m_cbIn = (uint8_t *)memalign(32, m_inImageDesc.m_width*m_inImageDesc.m_height*sizeof(uint8_t));
      m_crIn = (uint8_t *)memalign(32, m_inImageDesc.m_width*m_inImageDesc.m_height*sizeof(uint8_t));

      m_yOut  = (uint8_t *)memalign(32, m_outImageDesc.m_width*m_outImageDesc.m_height*sizeof(uint8_t));
      m_cbOut = (uint8_t *)memalign(32, m_outImageDesc.m_width*m_outImageDesc.m_height*sizeof(uint8_t));
      m_crOut = (uint8_t *)memalign(32, m_outImageDesc.m_width*m_outImageDesc.m_height*sizeof(uint8_t));

      m_cornersScore = (int16_t *)memalign(32, m_inImageDesc.m_width*m_inImageDesc.m_height*sizeof(int16_t));
      m_cornersMap   = (uint8_t *)memalign(32, m_inImageDesc.m_width*m_inImageDesc.m_height*sizeof(uint8_t));
      m_buffer       = (uint8_t *)memalign(32, m_inImageDesc.m_width*m_inImageDesc.m_height*sizeof(uint8_t));
     
      m_gX = (int16_t *) malloc(width * height * sizeof(int16_t));
      m_gY = (int16_t *) malloc(width * height * sizeof(int16_t));

      m_oldPyrBuf = (uint8_t *) malloc(width * height * 21 / 64);
      m_newPyrBuf = (uint8_t *) malloc(width * height * 21 / 64);

      m_oldPyr[0] = (uint8_t *) malloc(width * height);
      m_oldPyr[1] = m_oldPyrBuf;
      m_oldPyr[2] = m_oldPyrBuf + width / 2 * height / 2;
      m_oldPyr[3] = m_oldPyrBuf + width / 2 * height / 2 + width / 4 * height / 4;

//      m_newPyr[0] = inputImage;
      m_newPyr[1] = m_newPyrBuf;
      m_newPyr[2] = m_newPyrBuf + width / 2 * height / 2;
      m_newPyr[3] = m_newPyrBuf + width / 2 * height / 2 + width / 4 * height / 4;

      m_totalFeaturesNum = 100;
      m_currentFeaturesNum = 0;

      m_X    = (uint16_t *) malloc(m_totalFeaturesNum * sizeof(uint16_t));
      m_Y    = (uint16_t *) malloc(m_totalFeaturesNum * sizeof(uint16_t));

      memset(m_X, 0, m_totalFeaturesNum*sizeof(uint16_t));
      memset(m_Y, 0, m_totalFeaturesNum*sizeof(uint16_t));
      
      m_newX = (uint16_t *) malloc(m_totalFeaturesNum * sizeof(uint16_t));
      m_newY = (uint16_t *) malloc(m_totalFeaturesNum * sizeof(uint16_t));

      m_outError  = (uint16_t *) malloc(m_totalFeaturesNum * sizeof(uint16_t));

      m_pyramidX = (uint16_t *) malloc(m_totalFeaturesNum * sizeof(uint16_t));
      m_pyramidY = (uint16_t *) malloc(m_totalFeaturesNum * sizeof(uint16_t));
      m_scratch  = (uint8_t *) memalign(2, 893);      


      outTemp     = (uint16_t *) malloc(width*height * sizeof(uint16_t));
      pixIndex    = (int16_t *)  malloc((width*height*2 + 2) * sizeof(int16_t));
      internalBuf = (uint16_t *) malloc((width*height + 2*7) * sizeof(uint16_t));
      ind         = (int32_t *)  malloc(width*height * sizeof(int32_t));

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

#ifdef DEBUG_REPEAT
      for (unsigned repeat = 0; repeat < DEBUG_REPEAT; ++repeat) {
#endif
        if (m_inImageDesc.m_height > 0 && m_inImageDesc.m_width > 0) {
          //preprocessImage();
          YcbcrSeparation(_inImage);

          if (autoDetectHsv/* || m_currentFeaturesNum < m_totalFeaturesNum/2*/) {
            detectFeaturesToTrack(_inImage);
            setFeaturesToTrack();
          }

          doLKStuff(_inImage, _outImage);
          
          //postprocessImage();
        }
#ifdef DEBUG_REPEAT
      } // repeat
#endif

      for(int i = 0; i < 10; i++) {
        _outArgs.xs[i] = m_newX[i]>>4;
        _outArgs.ys[i] = m_newY[i]>>4;
      }
      _outArgs.xs[0] = good_points_number;
      return true;
    }
};

} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BALL_DETECTOR_SEQPASS_HPP_

