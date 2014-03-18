#ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_HPP_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_HPP_

#ifndef __cplusplus
#error C++-only header
#endif

#include <cassert>
#include <vector>
#include <c6x.h>

#include "internal/stdcpp.hpp"
#include "trik_vidtranscode_cv.h"


/* **** **** **** **** **** */ namespace trik /* **** **** **** **** **** */ {

/* **** **** **** **** **** */ namespace cv /* **** **** **** **** **** */ {


/*
class BitmapBuilder : public CVAlgorithm,
                      private assert_inst<false> // Generic instance, non-functional
{
  public:
    virtual bool setup(const TrikCvImageDesc& _inImageDesc,
                       const TrikCvImageDesc& _outImageDesc,
                       int8_t* _fastRam, size_t _fastRamSize) { return false; }
    virtual bool run(const TrikCvImageBuffer& _inImage, TrikCvImageBuffer& _outImage,
                     const TrikCvAlgInArgs& _inArgs, TrikCvAlgOutArgs& _outArgs);

  private:
};
*/

#if 0
#define DEBUG_INLINE __attribute__((noinline))
#else
#define DEBUG_INLINE __attribute__((always_inline))
#endif
//#define DEBUG_REPEAT 20


} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */

// include one of implementations
//#include "internal/cv_bitmap_bulder_seqpass.hpp"
#include "internal/cv_bitmap_builder_reference.hpp"

#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_BITMAP_BUILDER_HPP_
