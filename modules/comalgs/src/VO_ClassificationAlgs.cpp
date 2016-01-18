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
* Create Date:      2010-11-04                                              *
* Revise Date:      2012-03-22                                              *
*****************************************************************************/


#include <iostream>
#include <cstdio>
//#include "opencv/cv.h"
//#include "opencv/highgui.h"
#include "opencv2/highgui.hpp"
#include "VO_ClassificationAlgs.h"


/************************************************************************/
/*@author       JIA Pei                                                 */
/*@version      2010-11-04                                              */
/*@brief        Training                                                */
/*@param        data        Input - input data                          */
/*@param        categories  Input - a column vector                     */
/*@return       void                                                    */
/************************************************************************/
void CClassificationAlgs::Training( const Mat_<float>& data,
                                    const Mat_<int>& categories)
{
    unsigned int NbOfSamples = data.rows;
    set<int> ClassSet;
    for(int i = 0; i < categories.rows; i++)
    {
        ClassSet.insert(categories(i, 0));
    }
    this->m_iNbOfCategories = ClassSet.size();

    switch(this->m_iClassificationMethod)
    {
    case CClassificationAlgs::DecisionTree:
		this->m_CVDtree->setMaxDepth(INT_MAX);
		this->m_CVDtree->setMinSampleCount(2);
		this->m_CVDtree->setRegressionAccuracy(0.0f);
		this->m_CVDtree->setUseSurrogates(false);
		this->m_CVDtree->setMaxCategories(this->m_iNbOfCategories);
		this->m_CVDtree->setCVFolds(0);
		this->m_CVDtree->setUse1SERule(false);
		this->m_CVDtree->setTruncatePrunedTree(false);
		this->m_CVDtree->train(data, ml::SampleTypes::ROW_SAMPLE, categories);
        break;
    case CClassificationAlgs::Boost:
		this->m_CVBoost->setBoostType(ml::Boost::DISCRETE);
		this->m_CVBoost->setWeakCount(50);
		this->m_CVBoost->setWeightTrimRate(0.95);
		this->m_CVBoost->setMaxDepth(INT_MAX);
		this->m_CVBoost->setUseSurrogates(false);
		this->m_CVBoost->train(data, ml::SampleTypes::ROW_SAMPLE, categories);
        break;
    case CClassificationAlgs::RandomForest:
		this->m_CVRTrees->setMaxDepth(INT_MAX);
		this->m_CVRTrees->setMinSampleCount(2);
		this->m_CVRTrees->setRegressionAccuracy(0.0f);
		this->m_CVRTrees->setUseSurrogates(false);
		this->m_CVRTrees->setMaxCategories(this->m_iNbOfCategories);
		this->m_CVRTrees->setCalculateVarImportance(true);
		this->m_CVRTrees->setActiveVarCount(0);
		this->m_CVRTrees->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0.0f));
		this->m_CVRTrees->train(data, ml::SampleTypes::ROW_SAMPLE, categories);
		break;
	case CClassificationAlgs::ExtremeRandomForest:
		this->m_CVERTrees->setMaxDepth(INT_MAX);
		this->m_CVERTrees->setMinSampleCount(2);
		this->m_CVERTrees->setRegressionAccuracy(0.0f);
		this->m_CVERTrees->setUseSurrogates(false);
		this->m_CVERTrees->setMaxCategories(this->m_iNbOfCategories);
		this->m_CVERTrees->setCalculateVarImportance(true);
		this->m_CVERTrees->setActiveVarCount(0);
		this->m_CVERTrees->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0.0f));
		this->m_CVERTrees->train(data, ml::SampleTypes::ROW_SAMPLE, categories);
        break;
    case CClassificationAlgs::SVM:
		this->m_CVSVM->setType(ml::SVM::C_SVC);
		this->m_CVSVM->setKernel(ml::SVM::KernelTypes::RBF);
		this->m_CVSVM->setDegree(0);
		this->m_CVSVM->setGamma(1);
		this->m_CVSVM->setCoef0(0);
		this->m_CVSVM->setC(1);
		this->m_CVSVM->setNu(0);
		this->m_CVSVM->setP(0);
		this->m_CVSVM->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1E-6));
		this->m_CVSVM->train(data, ml::SampleTypes::ROW_SAMPLE, categories);
        break;
    }
}


/************************************************************************/
/*@author       JIA Pei                                                 */
/*@version      2010-11-04                                              */
/*@brief        Classification                                          */
/*@param        sample      Input - a sample to be classified           */
/*@return       the classified category                                 */
/************************************************************************/
int CClassificationAlgs::Classification(const Mat_<float>& sample )
{
    int res = -1;
    switch(this->m_iClassificationMethod)
    {
    case CClassificationAlgs::DecisionTree:
        {
            res = (int) this->m_CVDtree->predict( sample );
            //res = node->class_idx;
        }
        break;
    case CClassificationAlgs::Boost:
        {
            res = (int) this->m_CVBoost->predict( sample );
        }
        break;
    case CClassificationAlgs::RandomForest:
        {
            res = (int) this->m_CVRTrees->predict( sample );
        }
        break;
    case CClassificationAlgs::ExtremeRandomForest:
        {
            res = (int) this->m_CVERTrees->predict( sample );
        }
        break;
    case CClassificationAlgs::SVM:
    default:
        {
            res = (int) this->m_CVSVM->predict( sample );
        }
        break;
    }

    return res;
}


/** Save the classifier */
void CClassificationAlgs::Save(const string& fn ) const
{
    switch(this->m_iClassificationMethod)
    {
    case CClassificationAlgs::DecisionTree:
        {
            this->m_CVDtree->save(fn.c_str());
        }
        break;
    case CClassificationAlgs::Boost:
        {
            this->m_CVBoost->save(fn.c_str());
        }
        break;
    case CClassificationAlgs::RandomForest:
        {
            this->m_CVRTrees->save(fn.c_str());
        }
        break;
    case CClassificationAlgs::ExtremeRandomForest:
        {
            this->m_CVERTrees->save(fn.c_str());
        }
        break;
    case CClassificationAlgs::SVM:
    default:
        {
            this->m_CVSVM->save(fn.c_str());
        }
        break;
    }
}


/** Load the classifier */
void CClassificationAlgs::Load(const string& fn)
{
    switch(this->m_iClassificationMethod)
    {
    case CClassificationAlgs::DecisionTree:
        {
            this->m_CVDtree = Algorithm::load<ml::DTrees>(fn.c_str());
        }
        break;
    case CClassificationAlgs::Boost:
        {
            this->m_CVBoost = Algorithm::load<ml::Boost>(fn.c_str());
        }
        break;
    case CClassificationAlgs::RandomForest:
        {
            this->m_CVRTrees = Algorithm::load<ml::RTrees>(fn.c_str());
        }
        break;
    case CClassificationAlgs::ExtremeRandomForest:
        {
            this->m_CVERTrees = Algorithm::load<ml::RTrees>(fn.c_str());
        }
        break;
    case CClassificationAlgs::SVM:
        {
            this->m_CVSVM = Algorithm::load<ml::SVM>(fn.c_str());
        }
    case CClassificationAlgs::NONE:
    default:
        break;
    }
}

