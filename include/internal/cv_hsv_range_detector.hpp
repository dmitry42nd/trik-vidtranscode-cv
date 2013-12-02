#ifndef TRIK_VIDTRANSCODE_CV_INTERNAL_CV_HSV_RANGE_DETECTOR_SEQPASS_HPP_
#define TRIK_VIDTRANSCODE_CV_INTERNAL_CV_HSV_RANGE_DETECTOR_SEQPASS_HPP_

#ifndef __cplusplus
#error C++-only header
#endif

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <cassert>
#include <cmath>
#include <c6x.h>

#include "internal/stdcpp.hpp"


/* **** **** **** **** **** */ namespace trik /* **** **** **** **** **** */ {

/* **** **** **** **** **** */ namespace cv /* **** **** **** **** **** */ {

#warning Eliminate global var too

/*
//real clasters params
static const int cstrs_max_num = 32; // 256/8
static const int pos_shift = 3; // 8 == 2^3
*/

//test clasters params
static const int cstrs_max_num = 16; // 256/16
static const int pos_shift = 4; // 16 == 2^4

static int32_t s_hsv_clasters[cstrs_max_num][cstrs_max_num][cstrs_max_num];

class HsvRangeDetector
{
  private:
    static const int width = 320;
    static const int height = 240;
    
    //positive image part bounds
    static const int pos_l = 130;
    static const int pos_r = 190;
    static const int pos_t = 90;
    static const int pos_b = 150;

    //negative image part bounds
    static const int neg_l = 90;
    static const int neg_r = 230;
    static const int neg_t = 50;
    static const int neg_b = 190;

    //penalty coeffs
    static const int K0 = 2;
    static const int K1 = 1; 
    static const int K2 = 2; 

    const double T_end = 0.0005;

    const double e = 2.718281828;
    const double lambda = 0.76;

    union U_Hsv8x3
    {
        struct {
          uint8_t h;
          uint8_t s;
          uint8_t v;
          uint8_t none;
        } parts;
        uint32_t whole;
    };

    int getIncrement(int _val, int _min, int _max, double _t)
    {
        assert(_min <= _max);
        if(_min == _max)
        {
            return _min;
        }
        else
        {
            int res = 0;
            double base = 1 + 1/_t;
            double alpha = rand()/static_cast<double>(RAND_MAX);
            double degree = 2 * alpha - 1;

            res = _val + ((pow(base, degree) - 1) * _t)*static_cast<double>(_max - _min);

            if ((_min <= res) && (res < _max))
                return res;
            else
                return getIncrement(_val, _min, _max, _t);
        }
    }


    int truncateHue(int _val)
    {
        int res = _val % cstrs_max_num;
        if (res < 0)
            res += cstrs_max_num;
        return res;
    }

    uint64_t m_foo(int h1, int h2, int s1, int s2, int v1, int v2)
    {
        int64_t res = 0;
        int32_t val = 0;

        if (h1 <= h2)
        {

            for(int h_pos = h1; h_pos <= h2; h_pos++)
            {
                for(int s_pos = s1; s_pos <= s2; s_pos++)
                {
                    for(int v_pos = v1; v_pos <= v2; v_pos++)
                    {

                        val = s_hsv_clasters[h_pos][s_pos][v_pos];
                        res += val != 0 ? val : -K0;

                        //res += s_hsv_clasters[h_pos][s_pos][v_pos];
                    }
                }
            }

        }
        else // h1 > h2
        {
            for(int h_pos = h1; h_pos < cstrs_max_num; h_pos++)
            {
                for(int s_pos = s1; s_pos <= s2; s_pos++)
                {
                    for(int v_pos = v1; v_pos <= v2; v_pos++)
                    {

                        val = s_hsv_clasters[h_pos][s_pos][v_pos];
                        res += val != 0 ? val : -K0;

                        //res += s_hsv_clasters[h_pos][s_pos][v_pos];
                    }
                }
            }
            for(int h_pos = 0; h_pos <= h2; h_pos++)
            {
                for(int s_pos = s1; s_pos <= s2; s_pos++)
                {
                    for(int v_pos = v1; v_pos <= v2; v_pos++)
                    {

                        val = s_hsv_clasters[h_pos][s_pos][v_pos];
                        res += val != 0 ? val : -K0;

                    }
                }
            }
        }

        return res;
    }

  public:
    HsvRangeDetector()
    {}

    void detect(uint16_t& _h, uint16_t& _hTol, uint16_t& _s, uint16_t& _sTol, uint16_t& _v, uint16_t& _vTol, uint64_t* _rgb888hsv) 
    {
    //initialize stuff
      srand(time(NULL));
      const uint64_t* restrict img = _rgb888hsv;
      int h1;
      int h2;
      int s1;
      int s2;
      int v1;
      int v2;


    //initialize clasters
      memset(s_hsv_clasters, 0, cstrs_max_num*cstrs_max_num*cstrs_max_num*sizeof(int32_t));


    //initialize variables for claster with highest occurrence
      int h_max_pos = 0;
      int s_max_pos = 0;
      int v_max_pos = 0;
      int max_value = 0;

    //clasterize image
      U_Hsv8x3 pixel;
      int h_pos = 0;
      int s_pos = 0;
      int v_pos = 0;
      
      for (int row = 0; row < height; row++)
      {
        for (int col = 0; col < width; col++)
        {
          pixel.whole = _loll(*(img)++);
          h_pos = (pixel.parts.h >> pos_shift);
          s_pos = (pixel.parts.s >> pos_shift);
          v_pos = (pixel.parts.v >> pos_shift);

          //positive part of image
          if(pos_l < col && col < pos_r && pos_t < row && row < pos_b)
          {
            s_hsv_clasters[h_pos][s_pos][v_pos]+=K1;

            //remember claster with highest positive occurrence
            if (s_hsv_clasters[h_pos][s_pos][v_pos] > max_value)
            {
                max_value = s_hsv_clasters[h_pos][s_pos][v_pos];
                h_max_pos = h_pos;
                s_max_pos = s_pos;
                v_max_pos = v_pos;
            }

          } //negative part of image
          else if(neg_r < col || col < neg_l || neg_b < row || row < neg_t) //wrong condition!!
          {
            s_hsv_clasters[h_pos][s_pos][v_pos]-=K2;
          }
        }
      }

    //algorithm

      //initial hsv range
      h1 = h_max_pos;
      h2 = h_max_pos;
      s1 = s_max_pos;
      s2 = s_max_pos;
      v1 = v_max_pos;
      v2 = v_max_pos;

      int64_t L = m_foo(h1, h2, s1, s2, v1, v2);

      double T = 150;

      int h1New;
      int v1New;
      int h2New;
      int s1New;
      int s2New;
      int v2New;


      while(/*Th > T_end || Ts > T_end || Tv > T_end*/ T > T_end)
      {
#pragma MUST_ITERATE(200, ,200)
        for(int i = 0; i < 200; i++)
        {
          h1New = truncateHue(getIncrement(h1, 0, cstrs_max_num, T));
          h2New = truncateHue(getIncrement(h2, 0, cstrs_max_num, T));
          s1New = getIncrement(s1, 0, s_max_pos, T);
          s2New = getIncrement(s2, s_max_pos, cstrs_max_num, T);
          v1New = getIncrement(v1, 0, v_max_pos, T);
          v2New = getIncrement(v2, v_max_pos, cstrs_max_num, T);

          int64_t LNew = m_foo(h1New, h2New, s1New, s2New, v1New, v2New);

          if(L < LNew || (rand()/(double)RAND_MAX) <= pow(e,-(L-LNew)/T))
          {
            h1 = h1New;
            h2 = h2New;
            s1 = s1New;
            s2 = s2New;
            v1 = v1New;
            v2 = v2New;

            L = LNew;
          }
        }
        T*=lambda;
      }

      h1 = (h1 << pos_shift)*1.4f;
      h2 = (((h2+1) << pos_shift) - 1)*1.4f;

      s1 = (s1 << pos_shift)*0.39f;
      s2 = (((s2+1) << pos_shift) - 1)*0.39f;

      v1 = (v1 << pos_shift)*0.39f;
      v2 = (((v2+1) << pos_shift) - 1)*0.39f;

      if (h1 <= h2) 
      {
        _h    = (h2 + h1) / 2;
        _hTol = (h2 - h1) / 2;
      }
      else
      {
        float hue = (h2 - (360.0f - h1)) / 2;
        float hueTolerance = (h2 + (360.0f - h1)) / 2;
        _h = hue >= 0 ? hue : (hue + 360);
        _hTol = hueTolerance;
      }

      _s = (s2 + s1) / 2;
      _sTol = (s2 - s1) / 2;
      _v = (v1 + v2) / 2;
      _vTol = (v2 - v1) / 2;
    }

};


} /* **** **** **** **** **** * namespace cv * **** **** **** **** **** */

} /* **** **** **** **** **** * namespace trik * **** **** **** **** **** */


#endif // !TRIK_VIDTRANSCODE_CV_INTERNAL_CV_HSV_RANGE_DETECTOR_SEQPASS_HPP_
