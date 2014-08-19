#ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_STDCPP_H_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_STDCPP_H_

#ifndef __cplusplus
#error C++-only header
#endif

#include <cassert>


/* **** **** **** **** **** */ namespace trik /* **** **** **** **** **** */ {

#define IMG_WIDTH_MAX 640
#define IMG_HEIGHT_MAX 480

#define OBJECTS      4
#define METAPIX_SIZE 4

#define ENV_PIXS     4
#define NO_CLUSTER   0xFFFF


class noncopyable
{
  public:
    noncopyable() {}

  private:
    noncopyable(const noncopyable&);
    noncopyable& operator=(const noncopyable&);
};


template <bool _expr>
class assert_inst // kind of static_assert for TI compiler
{
  public:
    assert_inst()
    {
      char should_fail_on_false[_expr?1:-1]; // should not compile if _expr is false
      (void)should_fail_on_false; // warn prevention
    }

};


template <typename _T>
inline _T range(_T _min, _T _val, _T _max)
{
  if (_val < _min) return _min;
  else if (_val > _max) return _max;
  else return _val;
}


//Counting the number of non-zero bits in uint16_t variable
inline uint16_t pop(uint16_t x) 
{
  x = x - ((x >> 1) & 0x5555);
  x = (x & 0x3333) + ((x >> 2) & 0x3333);
  x = (x + (x >> 4)) & 0x0f0f;
  x = x + (x >> 8);
  x = x + (x >> 16);

  return x & 0x003f;
}

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_STDCPP_H_
