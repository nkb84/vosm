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


#include <iostream>
#include "VO_Triangle2DStructure.h"


ostream& operator<<(ostream& os, const VO_Triangle2DStructure& aamtriangle2d)
{
    os  << aamtriangle2d.GetVertexIndex(0) << " "
        << aamtriangle2d.m_MatShape(0,0) << " "
        << aamtriangle2d.m_MatShape(1,0)<< " "
        << aamtriangle2d.GetVertexIndex(1) << " "
        << aamtriangle2d.m_MatShape(0,1) << " "
        << aamtriangle2d.m_MatShape(1,1) << " "
        << aamtriangle2d.GetVertexIndex(2) << " "
        << aamtriangle2d.m_MatShape(0,2) << " "
        << aamtriangle2d.m_MatShape(1,2) << " "
        << aamtriangle2d.GetdD();

    return os;
}

istream& operator>>(istream &is, VO_Triangle2DStructure& aamtriangle2d)
{
    vector<unsigned int> vIndexes;
    Mat_<float> vVertexes = Mat_<float>::zeros(2,3);
    Point2f tmpPoint;
    int tempint;
    float tempdD;

    for(unsigned int j = 0; j < 3; j++)
    {
        is >> tempint;
        vIndexes.push_back(tempint);
        is >> tmpPoint.x >> tmpPoint.y;
        vVertexes(0,j) = tmpPoint.x;
        vVertexes(1,j) = tmpPoint.y;
    }
    is >> tempdD;
    aamtriangle2d.SetTheShape(vVertexes);
    aamtriangle2d.SetVertexIndexes(vIndexes);
    aamtriangle2d.SetdD( tempdD );

    return is;
}


/**
 * @brief       static function, a list of triangles to a shape
 * @param       triangles       -   Input    a list of triangles
 * @param       VO_Shape        -   Output   shape in Mat_<float>
 * @return      void
 */
VO_Shape VO_Triangle2DStructure::Triangle2D2Shape(const vector <VO_Triangle2DStructure>& triangles)
{
    vector<unsigned int> vi4oneTriangle;
    vector<unsigned int> countedIdx;
    vector<Point2f> ptList;

    for(unsigned int i = 0; i < triangles.size(); ++i)
    {
        vi4oneTriangle = triangles[i].GetVertexIndexes();

        for(unsigned int j = 0; j < 3; ++j)
        {
            if( !IsContaining(countedIdx, vi4oneTriangle[j]) )
            {
                countedIdx.push_back(vi4oneTriangle[j]);
                ptList.push_back( triangles[i].GetA2DPoint(j) );
            }
        }
    }
    
    Mat_<float> mat = Mat_<float>::zeros(2, ptList.size());
    
    for(unsigned int i = 0; i < ptList.size(); ++i)
    {
        mat(0, i) = ptList[i].x;
        mat(1, i) = ptList[i].y;
    }
    return mat;
}



/**
 * @brief        Adjust vertex sequence
 * @return       void
 */
void VO_Triangle2DStructure::AdjustVertexSequence()
{
    Point2f FirstVertex, SecondVertex, ThirdVertex;
    unsigned int FirstIndex, SecondIndex, ThirdIndex;

    if ( this->m_MatShape(1,0) <= this->m_MatShape(1,1))
    {
        if ( this->m_MatShape(1,0) <= this->m_MatShape(1,2))
        {
            FirstVertex    = this->GetA2DPoint(0);
            FirstIndex    = this->m_vVertexIndexes[0];
            if ( this->determinant( ) == COUNTER_CLOCKWISE )
            {
                SecondVertex    = this->GetA2DPoint(1);
                ThirdVertex        = this->GetA2DPoint(2);
                SecondIndex        = this->m_vVertexIndexes[1];
                ThirdIndex        = this->m_vVertexIndexes[2];
            }
            else
            {
                SecondVertex    = this->GetA2DPoint(2);
                ThirdVertex        = this->GetA2DPoint(1);
                SecondIndex        = this->m_vVertexIndexes[2];
                ThirdIndex        = this->m_vVertexIndexes[1];
            }
        }
        else
        {
            FirstVertex = this->GetA2DPoint(2);
            FirstIndex    = this->m_vVertexIndexes[2];
            if ( this->determinant( ) == COUNTER_CLOCKWISE )
            {
                SecondVertex    = this->GetA2DPoint(0);
                ThirdVertex        = this->GetA2DPoint(1);
                SecondIndex        = this->m_vVertexIndexes[0];
                ThirdIndex        = this->m_vVertexIndexes[1];
            }
            else
            {
                SecondVertex    = this->GetA2DPoint(1);
                ThirdVertex        = this->GetA2DPoint(0);
                SecondIndex        = this->m_vVertexIndexes[1];
                ThirdIndex        = this->m_vVertexIndexes[0];
            }
        }
    }
    else
    {
        if ( this->m_MatShape(1,1) <= this->m_MatShape(1,2))
        {
            FirstVertex = this->GetA2DPoint(1);
            FirstIndex    = this->m_vVertexIndexes[1];
            if ( this->determinant( ) == COUNTER_CLOCKWISE )
            {
                SecondVertex    = this->GetA2DPoint(2);
                ThirdVertex        = this->GetA2DPoint(0);
                SecondIndex        = this->m_vVertexIndexes[2];
                ThirdIndex        = this->m_vVertexIndexes[0];
            }
            else
            {
                SecondVertex    = this->GetA2DPoint(0);
                ThirdVertex        = this->GetA2DPoint(2);
                SecondIndex        = this->m_vVertexIndexes[0];
                ThirdIndex        = this->m_vVertexIndexes[2];
            }
        }
        else
        {
            FirstVertex = this->GetA2DPoint(2);
            FirstIndex    = this->m_vVertexIndexes[2];
            if ( this->determinant( ) == COUNTER_CLOCKWISE )
            {
                SecondVertex    = this->GetA2DPoint(0);
                ThirdVertex        = this->GetA2DPoint(1);
                SecondIndex        = this->m_vVertexIndexes[0];
                ThirdIndex        = this->m_vVertexIndexes[1];
            }
            else
            {
                SecondVertex    = this->GetA2DPoint(1);
                ThirdVertex        = this->GetA2DPoint(0);
                SecondIndex        = this->m_vVertexIndexes[1];
                ThirdIndex        = this->m_vVertexIndexes[0];
            }
        }
    }
    this->m_MatShape(0,0)        = FirstVertex.x;
    this->m_MatShape(1,0)        = FirstVertex.y;
    this->m_MatShape(0,1)        = SecondVertex.x;
    this->m_MatShape(1,1)        = SecondVertex.y;
    this->m_MatShape(0,2)        = ThirdVertex.x;
    this->m_MatShape(1,2)        = ThirdVertex.y;
    this->m_vVertexIndexes[0]    = FirstIndex;
    this->m_vVertexIndexes[1]    = SecondIndex;
    this->m_vVertexIndexes[2]    = ThirdIndex;
    this->Calc_dD();
}


/**
 * @author      JIA Pei
 * @version     2010-02-17
 * @brief       Judge whether one point is within a convex/concave hull of a shape
 * @param       pt              input        the concerned point
 * @param       triangles       input        all triangles composing the shape
 * @param       int             return the index of triangle containing pt
*/
int VO_Triangle2DStructure::IsPointInTriangles(const Point2f& pt, const vector<VO_Triangle2DStructure>& triangles)
{
    unsigned int NbOfTriangles  = triangles.size();
    Mat contour;

    for (unsigned int k = 0; k < NbOfTriangles; k++)
    {
        contour = triangles[k].ToPointList();

        // Here, we can actually give out all the local variables except the BGR,
        // the BGR is not shape based and will be given by function VO_LoadTexturesFromShapes()
        if (pointPolygonTest( contour, pt, false ) > 0.0)
        {
            return k;
        }
    }
    
    return -1;
}

