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
* Create Date:      2008-04-03                                              *
* Revise Date:      2012-03-22                                              *
*****************************************************************************/


#ifndef __VO_POINT2DDISTRIBUTIONMODEL_H__
#define __VO_POINT2DDISTRIBUTIONMODEL_H__


#include <vector>
//#include "opencv/cv.h"
//#include "opencv/highgui.h"
#include "opencv2/highgui.hpp"

#include "VO_Shape.h"
#include "VO_Ellipse.h"

using namespace std;
using namespace cv;


/** 
* @author   JIA Pei
* @brief    2D point distribution model
*/
class VO_Point2DDistributionModel
{
friend class VO_Fitting2DSM;
friend class VO_FittingAAMBasic;
friend class VO_FittingAAMForwardIA;
friend class VO_FittingAAMInverseIA;
friend class VO_FittingAFM;
friend class VO_FittingASMLTCs;
friend class VO_FittingASMNDProfiles;
protected:
    /** 2D normal distribution can be represented by rotated ellipse */
    vector<VO_Ellipse>          m_VONormalizedEllipses;

    /** Initialization */
    void init() {this->m_VONormalizedEllipses.clear();}

public:
    /** Default constructor to create a VO_Point2DDistributionModel object */
    VO_Point2DDistributionModel() {this->init();}

    /** Destructor */
    virtual ~VO_Point2DDistributionModel() {this->m_VONormalizedEllipses.clear();}

    /** Build Point Model */
    void                VO_BuildPointDistributionModel(    const vector<VO_Shape>& allAlignedShapes );

    /** Constrain respective single point according to its PDM */
    static void         VO_ConstrainSinglePoint(Point2f& pt, const VO_Ellipse& ellipse);
    
    /** Scale all ellipses in this PDM */
    static void         VO_ScalePDMEllipses(const vector<VO_Ellipse>& iEllipses, float scale, vector<VO_Ellipse>& oEllipses);

    /** Constrain a shape into Point Distribution Models */
    void                VO_ConstrainAllPoints(VO_Shape& ioShape);

    /** Save Point Model, to a specified folder */
    void                VO_Save(const string& fd);

    /** Load all parameters */
    void                VO_Load(const string& fd);

    /** Load parameters for fitting */
    void                VO_LoadParameters4Fitting(const string& fd);
    
    // Gets and Sets
    vector<VO_Ellipse>  GetPDMEllipses() const { return this->m_VONormalizedEllipses; }

};

#endif // __VO_POINT2DDISTRIBUTIONMODEL_H__

