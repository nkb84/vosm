/****************************************************************************
*                                                                           *
*   IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.       *
*                                                                           *
*   By downloading, copying, installing or using the software you agree to  *
*   this license. If you do not agree to this license, do not download,     *
*   install, copy or use the software.                                      *
*                                                                           *
*                           License Agreement                               *
*                   For Vision Open Statistical Models                      *
*                                                                           *
*   Copyright (C):      2006~2012 by JIA Pei, all rights reserved.          *
*                                                                           *
*   VOSM is free software under the terms of the GNU Lesser General Public  *
*   License (GNU LGPL) as published by the Free Software Foundation; either *
*   version 3.0 of the License, or (at your option) any later version.      *
*   You can use it, modify it, redistribute it, etc; and redistribution and *
*   use in source and binary forms, with or without modification, are       *
*   permitted provided that the following conditions are met:               *
*                                                                           *
*   a) Redistribution's of source code must retain this whole paragraph of  *
*   copyright notice, including this list of conditions and all the         *
*   following contents in this  copyright paragraph.                        *
*                                                                           *
*   b) Redistribution's in binary form must reproduce this whole paragraph  *
*   of copyright notice, including this list of conditions and all the      *
*   following contents in this copyright paragraph, and/or other materials  *
*   provided with the distribution.                                         *
*                                                                           *
*   c) The name of the copyright holders may not be used to endorse or      *
*   promote products derived from this software without specific prior      *
*   written permission.                                                     *
*                                                                           *
*   Any publications based on this code must cite the following five papers,*
*   technical reports and on-line materials.                                *
*   1) P. JIA, 2D Statistical Models, Technical Report of Vision Open       *
*   Working Group, 2st Edition, October 21, 2010.                           *
*   http://www.visionopen.com/members/jiapei/publications/pei_sm2dreport2010.pdf*
*   2) P. JIA. Audio-visual based HMI for an Intelligent Wheelchair.        *
*   PhD thesis, University of Essex, February, 2011.                        *
*   http://www.visionopen.com/members/jiapei/publications/pei_phdthesis2010.pdf*
*   3) T. Cootes and C. Taylor. Statistical models of appearance for        *
*   computer vision. Technical report, Imaging Science and Biomedical       *
*   Engineering, University of Manchester, March 8, 2004.                   *
*   http://www.isbe.man.ac.uk/~bim/Models/app_models.pdf                    *
*   4) I. Matthews and S. Baker. Active appearance models revisited.        *
*   International Journal of Computer Vision, 60(2):135--164, November 2004.*
*   http://www.ri.cmu.edu/pub_files/pub4/matthews_iain_2004_2/matthews_iain_2004_2.pdf*
*   5) M. B. Stegmann, Active Appearance Models: Theory, Extensions and     *
*   Cases, 2000.                                                            *
*   http://www2.imm.dtu.dk/~aam/main/                                       *
*                                                                           *
* Version:          0.4                                                     *
* Author:           JIA Pei                                                 *
* Contact:          jp4work@gmail.com                                       *
* URL:              http://www.visionopen.com                               *
* Create Date:      2010-03-05                                              *
* Revise Date:      2012-03-22                                              *
*****************************************************************************/

#ifndef __VO_STFT_H__
#define __VO_STFT_H__

#include <iostream>
//#include <opencv/cv.h>
//#include <opencv/highgui.h>
#include "opencv2/highgui.hpp"
#include "VO_WindowFunc.h"
#include "VO_IntegralTransform.h"


using namespace std;
using namespace cv;


/** 
 * @author    JIA Pei
 * @brief    Generalized class for short-time Fourier transform.
 */
class VO_STFT : public VO_IntegralTransform
{
protected:
    /**    This is used to record wavelet types. */
    VO_WindowFunc*      m_VOWindowFunc;

    /** Initialization */
    void                init()
    {
        this->m_VOWindowFunc    = NULL;
    }

public:
    /** Constructor */
    VO_STFT(Size windowSize = Size(0, 0), unsigned int windowFunc = 0)
    {
        this->init();
        this->m_VOWindowFunc    = new VO_WindowFunc(windowSize, windowFunc);
    }

    /** Destructor */
    ~VO_STFT()
    {
        if(this->m_VOWindowFunc)    delete this->m_VOWindowFunc;
    }

    /** prepare the STFT kernel */
    void                VO_PrepareKernel(unsigned int kernel = 0);

    /** Forward transform, must be Mat_<float> */
    void                VO_ForwardTransform(const Mat_<float>& iImg, Mat_<float>& oImg) const;

    /** Backward transform, must be Mat_<float> */
    void                VO_BackwardTransform(const Mat_<float>& iImg, Mat_<float>& oImg) const;

    /** Forward transform */
    void                VO_ForwardTransform(const Mat_<float>& iImg, Point pt, Mat_<float>& oImg) const;

    /** Get the window function */
    VO_WindowFunc*      GetWindowFunc() const {return this->m_VOWindowFunc;}
};

#endif // __VO_STFT_H__
