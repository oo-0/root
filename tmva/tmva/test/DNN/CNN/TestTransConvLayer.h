// @(#)root/tmva/tmva/cnn:$Id$
// Author: Ashish Kshirsagar

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Transpose convolution method on a CPU architecture                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Ashish Kshirsagar       <ashishkshirsagar10@gmail.com>                    *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

////////////////////////////////////////////////////////////////////
// Testing the Transpose Convolutional Layer                                //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;

inline bool isInteger(double x)
{
   return x == floor(x);
}

size_t calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   double dimension = ((imgDim - fltDim + 2 * padding) / stride) + 1;
   if (!isInteger(dimension)) {
      std::cout << "Not compatible hyper parameters" << std::endl;
      std::exit(EXIT_FAILURE);
   }

   return (size_t)dimension;
}

template<typename AFloat>
bool almostEqual(AFloat expected, AFloat computed, double epsilon = 0.0001) {
    return abs(computed - expected) < epsilon;
}

/*************************************************************************
 * Test 1: Forward Propagation
 *  batch size = 1
 *  image depth = 1, image height = 2, image width = 2,
 *  num frames = 1, filter height = 3, filter width = 3,
 *  stride rows = 1, stride cols = 1,
 *  zero-padding height = 0, zero-padding width = 0,
 *************************************************************************/
template<typename Architecture>
bool testForward1()
{
   using Matrix_t = typename Architecture::Matrix_t;

   double expected[][16] = {

    {
      2, 9, 6, 1, 
      6, 29, 30, 7,
      10, 29, 33, 13,
      12, 24, 16, 4
    }

   };

   double weights[][9] = {

    {
      1, 4, 1, 
      1, 4, 3,
      3, 3, 1
    }

   };

   double biases[][1] = {

      {
        0
      }

   };

   double img[][4] = {

      {
        2, 1, 4, 4
      }
    };
   
   size_t imgDepth = 2;
   size_t imgHeight = 2;
   size_t imgWidth = 2;
   size_t numberFilters = 2;
   size_t fltHeight = 3;
   size_t fltWidth = 3;
   size_t strideRows = 1;
   size_t strideCols = 1;
   size_t zeroPaddingHeight = 0;
   size_t zeroPaddingWidth = 0;

   Matrix_t inputEvent(imgDepth, imgHeight * imgWidth);

   for (size_t i = 0; i < imgDepth; i++) {
      for (size_t j = 0; j < imgHeight * imgWidth; j++) {
         inputEvent(i, j) = img[i][j];
      }
   }
   std::vector<Matrix_t> input;
   input.push_back(inputEvent);

   Matrix_t weightsMatrix(numberFilters, fltHeight * fltWidth * imgDepth);
   Matrix_t biasesMatrix(numberFilters, 1);
   for (size_t i = 0; i < numberFilters; i++) {
       for (size_t j = 0; j < fltHeight * fltWidth * imgDepth; j++){
           weightsMatrix(i, j) = weights[i][j];
       }
       biasesMatrix(i, 0) = biases[i][0];
   }

   size_t height = 4;//calculateDimension(imgHeight, fltHeight, zeroPaddingHeight, strideRows);
   size_t width = 4;//calculateDimension(imgWidth, fltWidth, zeroPaddingWidth, strideCols);

   Matrix_t outputEvent(numberFilters, height * width);

   for (size_t i = 0; i < numberFilters; i++) {
      for (size_t j = 0; j < height * width; j++) {
         outputEvent(i, j) = expected[i][j];
      }
   }
   
   std::cout<<"Expected Output Matrix "<<std::endl;
   for (size_t i = 0; i < numberFilters; i++) {
      for (size_t j = 0; j < height * width; j++) {
         std::cout<<outputEvent(i,j)<<" ";
      }
      std::cout<<std::endl;
   }   
   std::cout<<std::endl;

   std::vector<Matrix_t> expectedOutput;
   expectedOutput.push_back(outputEvent);
    std::cout<<std::endl;
    std::cout<<"Input image dimensions: "<<imgHeight<<" "<<imgWidth<<" "<<imgDepth<<std::endl;
    std::cout<<"Filter image dimensions: "<<fltHeight<<" "<<fltWidth<<" "<<numberFilters<<std::endl;
    std::cout<<"Stride "<<strideRows<<" "<<strideCols<<std::endl;
    std::cout<<"Padding "<<zeroPaddingHeight<<" "<<zeroPaddingWidth<<std::endl;


   std::cout<<"================================================================"<<std::endl;


   bool status = testTransConvLayerForward<Architecture>(input, expectedOutput, weightsMatrix, biasesMatrix, imgHeight,
                                                    imgWidth, imgDepth, fltHeight, fltWidth, numberFilters, strideRows,
                                                    strideCols, zeroPaddingHeight, zeroPaddingWidth);

   return status;
}