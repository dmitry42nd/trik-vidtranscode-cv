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

#include <ti/vlib/src/VLIB_goodFeaturestoTrack/VLIB_goodFeaturestoTrack.h>
#include <ti/vlib/src/VLIB_trackFeaturesLucasKanade_7x7/VLIB_trackFeaturesLucasKanade_7x7.h>
#include <ti/vlib/src/VLIB_imagePyramid8/VLIB_imagePyramid8.h>
#include <ti/vlib/src/VLIB_xyGradients/VLIB_xyGradients.h>

#include <ti/vlib/src/VLIB_xyGradientsAndMagnitude/VLIB_xyGradientsAndMagnitude.h>
#include <ti/vlib/src/VLIB_harrisScore_7x7/VLIB_harrisScore_7x7.h>
#include <ti/vlib/src/VLIB_nonMaxSuppress_7x7_S16/VLIB_nonMaxSuppress_7x7_S16.h>
}

#include "internal/stdcpp.hpp"
#include "trik_vidtranscode_cv.h"


/* **** **** **** **** **** */ namespace trik /* **** **** **** **** **** */ {

/* **** **** **** **** **** */ namespace cv /* **** **** **** **** **** */ {

static uint16_t s_rgb[320*240];
static uint8_t s_cb[320*240];
static uint8_t s_cr[320*240];

static uint32_t s_wi2wo[640];
static uint32_t s_hi2ho[480];

static uint16_t s_corners[320*240];

static int16_t s_xGrad[320*240+1];
static int16_t s_yGrad[320*240+1];

static uint16_t s_harrisScore[320*240+1];

static uint8_t s_buffer[200];

static const short s_coeff[5] = { 0x2000, 0x2BDD, -0x0AC5, -0x1658, 0x3770 };

template <>
class LKFeatureDetector<TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_YUV422P, TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_RGB565X> : public CVAlgorithm
{
  private:

    uint16_t harrisCornersNum;
   
    int16_t width;
    int16_t height;

    uint8_t *previousImage;
    uint8_t *inputImage;
    
    uint16_t *outError;

    int16_t *gradx;
    int16_t *grady;

    uint8_t *oldpyrbuf;
    uint8_t *newpyrbuf;

    uint8_t *oldpyr[3];
    uint8_t *newpyr[3];

    int32_t nFeatures;

    uint16_t *X;
    uint16_t *Y;

    uint16_t *newX;
    uint16_t *newY;

    int16_t *pyramidX;
    int16_t *pyramidY;
    uint8_t *scratch;
  
    double m_srcToDstShift;

    TrikCvImageDesc m_inImageDesc;
    TrikCvImageDesc m_outImageDesc;

    static uint16_t* restrict s_mult43_div;  // allocated from fast ram
    static uint16_t* restrict s_mult255_div; // allocated from fast ram
    
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

      const int32_t dstRow = s_hi2ho[srcRow];
      const int32_t dstCol = s_wi2wo[srcCol];

      const uint32_t dstOfs = dstRow*m_outImageDesc.m_lineLength + dstCol*sizeof(uint16_t);
      writeOutputPixel(reinterpret_cast<uint16_t*>(_outImage.m_ptr+dstOfs), _rgb888);
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
    
    
    void DEBUG_INLINE detectHarrisCorners(const TrikCvImageBuffer& _inImage)
    {
      const uint32_t width          = m_inImageDesc.m_width;
      const uint32_t height         = m_inImageDesc.m_height;

      //Harris corner detector
      const uint8_t* restrict in1 = reinterpret_cast<const uint8_t*>(_inImage.m_ptr);
      int16_t* restrict xgr = reinterpret_cast<int16_t*>(s_xGrad)+width+1;
      int16_t* restrict ygr = reinterpret_cast<int16_t*>(s_yGrad)+width+1;
      VLIB_xyGradients(in1, 
                       xgr, 
                       ygr,
                       width, height);

      int16_t* restrict hs1 = reinterpret_cast<int16_t*>(s_harrisScore);
      uint8_t* restrict buffer = reinterpret_cast<uint8_t*>(s_buffer);
      VLIB_harrisScore_7x7(xgr,
                           ygr,
                           width, height,
                           hs1,
                           3000, 
                           buffer);

      memset(reinterpret_cast<uint8_t*>(s_corners), 0, width*height * sizeof(uint8_t));
      memset(outTemp, 0, width*height * sizeof(uint16_t));


      VLIB_nonMaxSuppress_7x7_S16(reinterpret_cast<const int16_t*>(s_harrisScore), 
                                  width, height, 8000, 
                                  reinterpret_cast<uint8_t*>(s_corners));
      
    }

    
    void YcbcrSeparation(const TrikCvImageBuffer& _inImage) {
      //separate Cb Cr
      uint8_t* restrict cb   = reinterpret_cast<uint8_t*>(s_cb);
      uint8_t* restrict cr   = reinterpret_cast<uint8_t*>(s_cr);
      const uint16_t* restrict CbCr = reinterpret_cast<const uint16_t*>(_inImage.m_ptr +
                                                                        m_inImageDesc.m_lineLength*m_inImageDesc.m_height);
      #pragma MUST_ITERATE(8, ,8)
      for(int i = 0; i < width*height; i++) {
          *(cb++) = static_cast<uint8_t>(*CbCr);
          *(cr++) = static_cast<uint8_t>((*CbCr) >> 8);
          CbCr++;
      }
    }

    void setFeatures() {
      const uint32_t width  = m_inImageDesc.m_width;
      const uint32_t height = m_inImageDesc.m_height;
      int i = 0;
      
      memset(X, 0,nFeatures*sizeof(uint16_t));
      memset(Y, 0,nFeatures*sizeof(uint16_t));
      
    //lets try scaling out // & highlight corners
      const uint8_t* restrict corners  = reinterpret_cast<uint8_t*>(s_corners);
      #pragma MUST_ITERATE(8, ,8)
      for(int r = 0; r < height; r++) {
        #pragma MUST_ITERATE(8, ,8)
        for(int c = 0; c < width; c++) {
          if(c > 10 && c < 310 && r > 5 && r < 235)
            if (*corners != 0) {
              if (i < nFeatures) {
                X[i] = c<<4;
                Y[i] = r<<4;
                i++;
              }
              harrisCornersNum++;
            }
          corners++;
        }
      }
    }

    void doLKStuff(const TrikCvImageBuffer& _inImage, TrikCvImageBuffer& _outImage) {
      int i;
    
      // Obtain next frame, copy to block pointed by inputImage
      VLIB_imagePyramid8(reinterpret_cast<const uint8_t*>(_inImage.m_ptr), width, height, newpyrbuf);

      // Since we are starting from the level 3 in pyramid, the initial estimates of newX, newY are same as the
      // features of level 3 image of previousImage
      for(i = 0; i < nFeatures; i++) {
        newX[i] = X[i] >> 3;
        newY[i] = Y[i] >> 3;
      }

      // Update newX, newY three times starting from level 3 of pyramid up to level 1
      for (i = 3; i > 0; i--) {
        // pyramidX, pyramidY will have feature co-ordinates of level i of previousImage
        for (int j = 0; j < nFeatures; j++) {
          pyramidX[j] = X[j] >> i;
          pyramidY[j] = Y[j] >> i;
        }

        // Estimates newX, newY are updated at level i. Input features are pyramidX, pyramidY
        VLIB_xyGradients(oldpyr[i], gradx + (width >> i) + 1, grady + (width >> i) + 1, width >> i, height >> i);
        VLIB_trackFeaturesLucasKanade_7x7(oldpyr[i], newpyr[i], 
                                          gradx, grady, 
                                          width >> i, height >> i, 
                                          nFeatures,
                                          reinterpret_cast<const uint16_t*>(pyramidX), reinterpret_cast<const uint16_t*>(pyramidY), 
                                          newX, newY, 
                                          outError, 10, 0, scratch);

        // newX, newY refined at level i are scaled to become estimates for next iteration
        for (int j = 0; j < nFeatures; j++) {
          newX[j] = newX[j] << 1;
          newY[j] = newY[j] << 1;
        }
      }

      //Fine tune newX, newY fourth time with original resolution images
      int16_t* restrict xgr = reinterpret_cast<int16_t*>(s_xGrad);
      int16_t* restrict ygr = reinterpret_cast<int16_t*>(s_yGrad);      
      VLIB_xyGradients(previousImage, xgr+width+1, ygr+width+1, width, height);
      VLIB_trackFeaturesLucasKanade_7x7(previousImage, reinterpret_cast<const uint8_t*>(_inImage.m_ptr), 
                                        xgr, ygr, 
                                        width, height,
                                        nFeatures,
                                        reinterpret_cast<const uint16_t*>(X), reinterpret_cast<const uint16_t*>(Y), 
                                        newX, newY,
                                        outError, 10, 0, scratch);
      // (newX[i]>>4, newY[i]>>4) will now have pixel level feature co-ordinates for inputImage for all 0 <= i < nFeatures

      // make inputImage, its pyramid and features 'old' for next iteration
      memcpy(previousImage, _inImage.m_ptr, width * height);
      memcpy(oldpyrbuf, newpyrbuf, width * height * 21 / 64);
      for(i = 0; i < nFeatures; i++){
        X[i] = newX[i];
        Y[i] = newY[i];
      }    

      //in_img to rgb565
      const short* restrict coeff = s_coeff;
      const unsigned char* restrict y_in  = reinterpret_cast<const unsigned char*>(_inImage.m_ptr);
      const unsigned char* restrict cb_in = reinterpret_cast<const unsigned char*>(s_cb);
      const unsigned char* restrict cr_in = reinterpret_cast<const unsigned char*>(s_cr);
      unsigned short* rgb565_out          = reinterpret_cast<unsigned short*>(s_rgb);
      IMG_ycbcr422pl_to_rgb565(coeff, y_in, cb_in, cr_in, rgb565_out, width*height);

      //lets try scaling out // & highlight corners
      const uint16_t* restrict imgRgb565ptr  = reinterpret_cast<uint16_t*>(s_rgb);
      const uint8_t* restrict corners  = reinterpret_cast<uint8_t*>(s_corners);
      const uint32_t dstLineLength  = m_outImageDesc.m_lineLength;
      const uint32_t* restrict p_hi2ho = s_hi2ho;
      #pragma MUST_ITERATE(8, ,8)
      for(int r = 0; r < height; r++) {
        const uint32_t dR = *(p_hi2ho++);
        uint16_t* restrict dIR = reinterpret_cast<uint16_t*>(_outImage.m_ptr + dR*dstLineLength);
        const uint32_t* restrict p_wi2wo = s_wi2wo;
        #pragma MUST_ITERATE(8, ,8)
        for(int c = 0; c < width; c++) {
          const uint32_t dC    = *(p_wi2wo++);
          *(dIR+dC) = *(imgRgb565ptr++);
            
          if(c > 10 && c < 310 && r > 5 && r < 235)
            if (*corners != 0) {
              drawCornerHighlight(c, r, _outImage, 0xff0000);
            }
          corners++;
        }
      }
        
      for (i = 0; i < nFeatures; i++)
        drawCornerHighlight(newX[i]>>4, newY[i]>>4, _outImage, 0x00ff00);
    
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
      const double srcToDstShift = min(static_cast<double>(m_outImageDesc.m_width)/m_inImageDesc.m_width, 
                                 static_cast<double>(m_outImageDesc.m_height)/m_inImageDesc.m_height);

      const uint32_t widthIn  = _inImageDesc.m_width;
      uint32_t* restrict p_wi2wo = s_wi2wo;
      for(int i = 0; i < widthIn; i++) {
          *(p_wi2wo++) = i*srcToDstShift;
      }

      const uint32_t heightIn  = _inImageDesc.m_height;
      uint32_t* restrict p_hi2ho = s_hi2ho;
      for(uint32_t i = 0; i < heightIn; i++) {
          *(p_hi2ho++) = i*srcToDstShift;
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
      
      width          = m_inImageDesc.m_width;
      height         = m_inImageDesc.m_height;

      previousImage = (uint8_t *) malloc(width * height);
      inputImage    = (uint8_t *) malloc(width * height);

      gradx = (int16_t *) malloc(width * height * sizeof(int16_t));
      grady = (int16_t *) malloc(width * height * sizeof(int16_t));

      oldpyrbuf = (uint8_t *) malloc(width * height * 21 / 64);
      newpyrbuf = (uint8_t *) malloc(width * height * 21 / 64);

      oldpyr[0] = previousImage;
      oldpyr[1] = oldpyrbuf;
      oldpyr[2] = oldpyrbuf + width / 2 * height / 2;
      oldpyr[3] = oldpyrbuf + width / 2 * height / 2 + width / 4 * height / 4;

      newpyr[0] = inputImage;
      newpyr[1] = newpyrbuf;
      newpyr[2] = newpyrbuf + width / 2 * height / 2;
      newpyr[3] = newpyrbuf + width / 2 * height / 2 + width / 4 * height / 4;

      nFeatures = 200;
      harrisCornersNum = 0;

      X    = (uint16_t *) malloc(nFeatures * sizeof(uint16_t));
      Y    = (uint16_t *) malloc(nFeatures * sizeof(uint16_t));

      memset(X,0,nFeatures * sizeof(uint16_t));
      memset(Y,0,nFeatures * sizeof(uint16_t));
      
      newX = (uint16_t *) malloc(nFeatures * sizeof(uint16_t));
      newY = (uint16_t *) malloc(nFeatures * sizeof(uint16_t));

      outError  = (uint16_t *) malloc(nFeatures * sizeof(uint16_t));

      pyramidX = (int16_t *) malloc(nFeatures * sizeof(int16_t));
      pyramidY = (int16_t *) malloc(nFeatures * sizeof(int16_t));
      scratch  = (uint8_t *) memalign(2, 893);      
      
      //VLIB_imagePyramid8(previousImage, width, height, oldpyrbuf);
      
      
      outTemp     =  (uint16_t *) malloc(width*height * sizeof(uint16_t));
      pixIndex    =  (int16_t *)  malloc((width*height*2 + 2) * sizeof(int16_t));
      internalBuf =  (uint16_t *) malloc((width*height + 2*7) * sizeof(uint16_t));
      ind         =  (int32_t *)  malloc(width*height * sizeof(int32_t));
      
      
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
          YcbcrSeparation(_inImage);
          if (autoDetectHsv) {
            detectHarrisCorners(_inImage);
            setFeatures();
          }
          doLKStuff(_inImage, _outImage);
        }
#ifdef DEBUG_REPEAT
      } // repeat
#endif

      for(int i = 0; i < 10; i++) {
        _outArgs.xs[i] = newX[i]>>4;
        _outArgs.ys[i] = newY[i]>>4;
      }
      
      _outArgs.xs[0] = harrisCornersNum;
      
      return true;
    }
};

uint16_t* restrict LKFeatureDetector<TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_YUV422P,
                                TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_RGB565X>::s_mult43_div = NULL;
uint16_t* restrict LKFeatureDetector<TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_YUV422P,
                                TRIK_VIDTRANSCODE_CV_VIDEO_FORMAT_RGB565X>::s_mult255_div = NULL;


} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BALL_DETECTOR_SEQPASS_HPP_

