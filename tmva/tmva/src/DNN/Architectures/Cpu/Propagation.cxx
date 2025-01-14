// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
// Implementation of the functions required for the forward and     //
// backward propagation of activations through a neural network for //
// the reference implementation.                                    //
//////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TMVA/DNN/Architectures/Cpu/Blas.h"

namespace TMVA {
namespace DNN {

template <typename AFloat>
void TCpu<AFloat>::MultiplyTranspose(TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &input,
                                     const TCpuMatrix<AFloat> &Weights)
{

   int m = (int)input.GetNrows();
   int k = (int)input.GetNcols();
   int n = (int)Weights.GetNrows();

   if ((int)output.GetNrows() != m) {
      Error("MultiplyTranspose","Invalid input - output  rows  - input:  %d != output : %d",m, (int) output.GetNrows());
      R__ASSERT((int) output.GetNrows() == m);
   }
   if ((int)output.GetNcols() != n) {
      Error("MultiplyTranspose","Invalid output cols or weight  rows  - output cols:  %d != weight rows : %d",(int) output.GetNcols(),n);
      R__ASSERT((int) output.GetNcols() == n);
   }
   if ((int)Weights.GetNcols() != k) {
      Error("MultiplyTranspose","Invalid input cols or weight cols  - input cols:  %d != weight cols : %d", k, (int) Weights.GetNcols());
      R__ASSERT((int) Weights.GetNcols() == k); 
   }

   char transa = 'N';
   char transb = 'T';

   AFloat alpha = 1.0;
   AFloat beta = 0.0;

   const AFloat *A = input.GetRawDataPointer();
   const AFloat *B = Weights.GetRawDataPointer();
   AFloat *C = output.GetRawDataPointer();

   ::TMVA::DNN::Blas::Gemm(&transa, &transb, &m, &n, &k, &alpha, A, &m, B, &n, &beta, C, &m);
}

template <typename AFloat>
void TCpu<AFloat>::AddRowWise(TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &biases)
{
   int m = (int)output.GetNrows();
   int n = (int)output.GetNcols();

   int inc = 1.0;
   AFloat alpha = 1.0;

   AFloat *A = output.GetRawDataPointer();
   const AFloat *x = TCpuMatrix<AFloat>::GetOnePointer();
   const AFloat *y = biases.GetRawDataPointer();

   R__ASSERT(m <= (int)TCpuMatrix<AFloat>::GetOnePointerSize()); 
   R__ASSERT(n <= (int)(biases.GetNcols()*biases.GetNrows())); 

   ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);
}

template <typename AFloat>
void TCpu<AFloat>::Backward(TCpuMatrix<AFloat> &activationGradientsBackward, TCpuMatrix<AFloat> &weightGradients,
                            TCpuMatrix<AFloat> &biasGradients, TCpuMatrix<AFloat> &df,
                            const TCpuMatrix<AFloat> &activationGradients, const TCpuMatrix<AFloat> &weights,
                            const TCpuMatrix<AFloat> &activationsBackward)
{
   // Compute element-wise product.
   Hadamard(df, activationGradients);

   // Activation gradients.
   if (activationGradientsBackward.GetNoElements() > 0) Multiply(activationGradientsBackward, df, weights);

   // Weight gradients.
   if (weightGradients.GetNoElements() > 0) TransposeMultiply(weightGradients, df, activationsBackward);

   // Bias gradients.
   if (biasGradients.GetNoElements() > 0) SumColumns(biasGradients, df);
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Im2col(TCpuMatrix<AFloat> &A, const TCpuMatrix<AFloat> &B, size_t imgHeight, size_t imgWidth,
                          size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols,
                          size_t zeroPaddingHeight, size_t zeroPaddingWidth)
{

   // image boudaries
   int imgHeightBound = imgHeight + zeroPaddingHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth + zeroPaddingWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;

   const int halfFltHeight =  fltHeight / 2;
   const int halfFltWidth =  fltWidth / 2;
   const int halfFltHeightM1 = (fltHeight - 1) / 2;
   const int halfFltWidthM1 = (fltWidth - 1) / 2;
   const int nRowsInput = B.GetNrows();
   const int nColsInput = B.GetNcols(); 
   const int nRowsOutput = A.GetNrows();
   const int nColsOutput = A.GetNcols(); 

   // convolution centers
   for (int i = halfFltHeight -zeroPaddingHeight; i <= imgHeightBound; i += strideRows) {
      for (int j = halfFltWidth -zeroPaddingWidth ; j <= imgWidthBound; j += strideCols) {
         size_t currLocalViewPixel = 0;

         // within the local view
         R__ASSERT((int) currLocalView < nRowsOutput );

         for (int m = 0; m < nRowsInput; m++) {
            for (int k = i - halfFltHeight  ; k <= Int_t(i + halfFltHeightM1 ); k++) {
               int kstep = k * imgWidth;
               for (int l = j - halfFltWidth ; l <= Int_t(j + halfFltWidthM1); l++) {

                  // Check the boundaries
                  R__ASSERT((int) currLocalViewPixel < nColsOutput );
                  //R__ASSERT(k * imgWidth + l < B.GetNcols());
                  if (k < 0 || k >= (Int_t)imgHeight || l < 0 || l >= (Int_t)imgWidth || kstep + l >=  nColsInput)
                     A(currLocalView, currLocalViewPixel++) = 0;
                  else
                     A(currLocalView, currLocalViewPixel++) = B(m, kstep + l);
               }
            }
         }
         //std::cout << " i " << i << "  " << j << " increment currLocalView " << currLocalView << std::endl;
         currLocalView++;
      }
   }
   //TMVA_DNN_PrintTCpuMatrix(A,"FromIm2Col"); 
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Im2colIndices(std::vector<int> &V, const TCpuMatrix<AFloat> &B, size_t nLocalViews, size_t imgHeight, size_t imgWidth,
                          size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols,
                           size_t zeroPaddingHeight, size_t zeroPaddingWidth)
{

   // image boudaries
   int imgHeightBound = imgHeight + zeroPaddingHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth + zeroPaddingWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;

   const int halfFltHeight =  fltHeight / 2;
   const int halfFltWidth =  fltWidth / 2;
   const int halfFltHeightM1 = (fltHeight - 1) / 2;
   const int halfFltWidthM1 = (fltWidth - 1) / 2;
   const int nRowsInput = B.GetNrows();
   const int nColsInput = B.GetNcols();
   const size_t nSizeOutput = V.size();
   const int npixels =  nRowsInput * fltHeight * fltWidth;
   // const int nRowsOutput = A.GetNrows();
   // const int nColsOutput = A.GetNcols(); 

   // convolution centers
   for (int i = halfFltHeight -zeroPaddingHeight; i <= imgHeightBound; i += strideRows) {
      for (int j = halfFltWidth -zeroPaddingWidth ; j <= imgWidthBound; j += strideCols) {
         size_t currLocalViewPixel = 0;

         // within the local view
         //R__ASSERT((int) currLocalView < nRowsOutput );

         for (int m = 0; m < nRowsInput; m++) {
            for (int k = i - halfFltHeight  ; k <= Int_t(i + halfFltHeightM1 ); k++) {
               int kstep = k * imgWidth;
               for (int l = j - halfFltWidth ; l <= Int_t(j + halfFltWidthM1); l++) {

                  // Check the boundaries
                  //R__ASSERT(currLocalViewPixel < nColsOutput );
                  R__ASSERT(currLocalView * npixels + currLocalViewPixel < nSizeOutput ); 
                  if (k < 0 || k >= (Int_t)imgHeight || l < 0 || l >= (Int_t)imgWidth || kstep + l >=  nColsInput)
                     //V[currLocalView * npixels + currLocalViewPixel]=-1;
                     V[currLocalViewPixel * nLocalViews + currLocalView] = -1;
                  else
                     V[currLocalViewPixel * nLocalViews + currLocalView]= ( kstep + l) * nRowsInput + m;
                  
                  currLocalViewPixel++;
               }
            }
         }
         currLocalView++;
      }
   }
}
template <typename AFloat>
void TCpu<AFloat>::Im2colFast(TCpuMatrix<AFloat> &A, const TCpuMatrix<AFloat> &B, const std::vector<int> &V) 
{
   size_t  n = V.size(); 
   R__ASSERT( n == A.GetNcols() * A.GetNrows() );
   AFloat *  a = A.GetRawDataPointer();
   const AFloat *  b = B.GetRawDataPointer();

//#define DL_USE_MTE  
   // parallel execution
#ifdef DL_USE_MTE
   const size_t nsteps = TCpuMatrix<AFloat>::GetNWorkItems(n);

   auto f = [&](UInt_t workerID)
   {
      for (size_t j = 0; j < nsteps; ++j) {
         size_t ii = workerID+j;
         if (ii >= n) break;
         int idx = V[ii]; 
         if (idx >= 0) a[ii] = b[idx];
         else a[ii] = 0;
      }
      return 0;
   };

   A.GetThreadExecutor().Foreach(f, ROOT::TSeqI(0,n,nsteps) );

#else
   //serial execution
   for (size_t ii = 0; ii < n; ++ii) {
      int idx = V[ii]; 
      if (idx >= 0) a[ii] = b[idx];
      else a[ii] = 0;
   }

#endif
   // TMVA_DNN_PrintTCpuMatrix(A,"FromFastIm2Col");
   // TMVA_DNN_PrintTCpuMatrix(B,"input to Im2Col");
   // std::cout << "V vector " << V.size() << std::endl;
   // for ( int i = 0; i < n; ++i) {
   //    std::cout << V[i] << "  ";
   // }
   // std::cout << std::endl;
}
//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::RotateWeights(TCpuMatrix<AFloat> &A, const TCpuMatrix<AFloat> &B, size_t filterDepth,
                                 size_t filterHeight, size_t filterWidth, size_t numFilters)
{
   size_t jump = filterHeight * filterWidth;
   for (size_t j = 0; j < filterDepth; j++) {
      for (size_t k = 0; k < numFilters; k++) {
         for (size_t i = 0; i < jump; i++) {
            A(j, k * jump + i) = B(k, ((j + 1) * jump - 1) - i);
            //A(j, k * jump + i) = B(k, j * jump + i);
         }
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::AddConvBiases(TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &biases)
{
   int m = (int)output.GetNrows();
   int n = (int)output.GetNcols();

   int inc = 1.0;
   AFloat alpha = 1.0;

   AFloat *A = output.GetRawDataPointer();
   const AFloat *x = biases.GetRawDataPointer();
   const AFloat *y = TCpuMatrix<AFloat>::GetOnePointer();

   R__ASSERT(m <= (int)biases.GetNoElements() ); 
   R__ASSERT(n <= (int)TCpuMatrix<AFloat>::GetOnePointerSize() ); 

   ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);
}

template<typename AFloat>
size_t TCpu<AFloat>::calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   size_t temp = imgDim - fltDim + 2 * padding;
   if (temp % stride || temp + stride <= 0) {
      Fatal("calculateDimension", "Not compatible hyper parameters for layer - (imageDim, filterDim, padding, stride) "
            "%zu, %zu, %zu, %zu", imgDim, fltDim, padding, stride);
   }
   return temp / stride + 1;
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::ConvLayerForward(std::vector<TCpuMatrix<AFloat>> & output,
                                    std::vector<TCpuMatrix<AFloat>> & derivatives,
                                    const std::vector<TCpuMatrix<AFloat>> &input,
                                    const TCpuMatrix<AFloat> &weights, const TCpuMatrix<AFloat> & biases,
                                    const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                    std::vector<TCpuMatrix<AFloat>> & /*  */)
{
   size_t height = calculateDimension(params.inputHeight, params.filterHeight, params.paddingHeight, params.strideRows);
   size_t width = calculateDimension(params.inputWidth, params.filterWidth, params.paddingWidth, params.strideCols);
   size_t nLocalViews = height * width;
   size_t nLocalViewPixels = params.inputDepth * params.filterHeight * params.filterWidth;

   R__ASSERT( input.size() > 0);
   std::vector<int> forwardIndices(nLocalViews * nLocalViewPixels);
   Im2colIndices(forwardIndices, input[0], nLocalViews, params.inputHeight, params.inputWidth, params.filterHeight,
                 params.filterWidth, params.strideRows, params.strideCols, params.paddingHeight, params.paddingWidth);

   //this should fix multi-thread inizializations of arrays
   TCpuMatrix<AFloat>::InitializeOneVector(nLocalViews);
   TCpuMatrix<AFloat>::InitializeOneVector(output[0].GetNcols());   // since it is used in AddCOnvBiases


   auto f = [&] (UInt_t i)
   {
       // dropout not yet implemented for CNN
       // if (applyDropout && (dropoutProbability != 1.0)) {
       //    Dropout(input[i], dropoutProbability);
       // }

       TCpuMatrix<AFloat> inputTr(nLocalViews, nLocalViewPixels);
       //inputTr.Zero();   // this is not thread safe

       Im2colFast(inputTr, input[i], forwardIndices);

       MultiplyTranspose(output[i], weights, inputTr);
       AddConvBiases(output[i], biases);

       evaluateDerivative<TCpu<AFloat>>(derivatives[i], activFunc, output[i]);
       evaluate<TCpu<AFloat>>(output[i], activFunc);

   };

   TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(f, ROOT::TSeqI(input.size()));
}

template <typename AFloat>
void TCpu<AFloat>::GenerateConvMatrix(TCpuMatrix<AFloat> weights, 
                                    std::vector< TCpuMatrix<AFloat> > & modifiedWeightMatrix){
  //TCpuMatrix<AFloat> modifiedWeightMatrix(rows,cols);
  size_t rows = modifiedWeightMatrix[0].GetNrows();
  size_t cols = modifiedWeightMatrix[0].GetNcols();
  TCpuMatrix<AFloat> columnarWeightMatrix(weights.GetNrows()*weights.GetNcols(),1);
  std::vector< TCpuMatrix<AFloat> > columnarVector;
  columnarVector.emplace_back(columnarWeightMatrix);
  GenerateColumnarMatrix(weights,columnarVector);
  size_t padRow = 0;
  for(size_t l = 0; l < columnarVector[0].GetNrows();l++){
    std::cout<<columnarVector[0](l,0)<<" ";
  }
  std::cout<<std::endl;
  std::cout<<"Weight Matrix "<<weights.GetNrows()<<" "<<weights.GetNcols()<<std::endl;
  std::cout<<"Modified Weight Matrix "<<std::endl;
  for(size_t i = 0 ; i < cols;){
    for(size_t a = 0; a < rows; a++){
      for(size_t b = 0; b < cols; b++){
        std::cout<<modifiedWeightMatrix[0](a,b)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
    size_t j = 0;
    while(j < rows){
      size_t count = 1;
      for(size_t k = 0; k < padRow; k++){
        modifiedWeightMatrix[0](k,i) = 0 ;
        j++;
      }
      size_t weightIndex = 0;
      for( ; weightIndex < columnarVector[0].GetNrows() and j<rows;){
        
        std::cout<<"( "<<j<<", "<<i<<" ), count = "<<count<<" , wI = "<<weightIndex<<std::endl;
        std::cout<<"Accessing modifiedWeightMatrix( "<<j<<", "<<i<<" ) and columnarVector( "<<weightIndex<<", "<<0<<" )"<<std::endl;
        std::cout<<"Capacity for both are ( "<<modifiedWeightMatrix[0].GetNrows()<<", "<<modifiedWeightMatrix[0].GetNcols()<<" ) and ( "<<columnarVector[0].GetNrows()<<", "<<columnarVector[0].GetNcols()<<" ) "<<std::endl;
        if(count%(weights.GetNcols()+1)==0){
          modifiedWeightMatrix[0](j,i) = 0;
        }
        else{
          modifiedWeightMatrix[0](j,i) = columnarVector[0](weightIndex,0);  
          weightIndex+=1;
        }
        j++;
        count +=1;
      }
      
      for(size_t k = j; k < rows; k++)
      {
        modifiedWeightMatrix[0](j,i) = 0;
        j++;
      }
    } 
    i++;
    std::cout<<"Padding "<<padRow<<" -> ";
    if(i >= (cols/2)){
      int newPadRow =  (rows - 1 - columnarVector[0].GetNrows() - (weights.GetNrows() - 1)) - (cols - 1 - i);
      if(newPadRow<0){
        newPadRow = 0;
      }
      padRow = newPadRow;
    }
    else{
      padRow = i ;
    }
    std::cout<<padRow<<std::endl;
  }
  //return modifiedWeightMatrix;
}

template <typename AFloat>
void TCpu<AFloat>::GenerateTransConvMatrix(TCpuMatrix<AFloat> weights, 
                                    std::vector< TCpuMatrix<AFloat> > & modifiedWeightMatrix){
  //TCpuMatrix<AFloat> modifiedWeightMatrix(rows,cols);
  size_t rows = modifiedWeightMatrix[0].GetNrows();
  size_t cols = modifiedWeightMatrix[0].GetNcols();
  TCpuMatrix<AFloat> columnarWeightMatrix(weights.GetNrows()*weights.GetNcols(),1);
  std::vector< TCpuMatrix<AFloat> > columnarVector;
  columnarVector.emplace_back(columnarWeightMatrix);
  GenerateColumnarMatrix(weights,columnarVector);
  size_t padCol = 0;
  for(size_t l = 0; l < columnarVector[0].GetNrows();l++){
    std::cout<<columnarVector[0](l,0)<<" ";
  }
  std::cout<<std::endl;
  std::cout<<"Weight Matrix "<<weights.GetNrows()<<" "<<weights.GetNcols()<<std::endl;
  std::cout<<"Modified Weight Matrix "<<std::endl;
  for(size_t i = 0 ; i < rows;){
    for(size_t a = 0; a < rows; a++){
      for(size_t b = 0; b < cols; b++){
        std::cout<<modifiedWeightMatrix[0](a,b)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
    size_t j = 0;
    while(j < cols){
      size_t count = 1;
      for(size_t k = 0; k < padCol; k++){
        modifiedWeightMatrix[0](i,k) = 0 ;
        j++;
      }
      size_t weightIndex = 0;
      for( ; weightIndex < columnarVector[0].GetNrows() and j<cols;){
        
        std::cout<<"( "<<j<<", "<<i<<" ), count = "<<count<<" , wI = "<<weightIndex<<std::endl;
        std::cout<<"Accessing modifiedWeightMatrix( "<<j<<", "<<i<<" ) and columnarVector( "<<weightIndex<<", "<<0<<" )"<<std::endl;
        std::cout<<"Capacity for both are ( "<<modifiedWeightMatrix[0].GetNrows()<<", "<<modifiedWeightMatrix[0].GetNcols()<<" ) and ( "<<columnarVector[0].GetNrows()<<", "<<columnarVector[0].GetNcols()<<" ) "<<std::endl;
        if(count%(weights.GetNcols()+1)==0){
          modifiedWeightMatrix[0](i,j) = 0;
        }
        else{
          modifiedWeightMatrix[0](i,j) = columnarVector[0](weightIndex,0);  
          weightIndex+=1;
        }
        j++;
        count +=1;
      }
      
      for(size_t k = j; k < cols; k++)
      {
        modifiedWeightMatrix[0](i,j) = 0;
        j++;
      }
    } 
    i++;
    std::cout<<"Padding "<<padCol<<" -> ";
    if(i >= (rows/2)){
      int newPadCol =  (cols - 1 - columnarVector[0].GetNrows() - (weights.GetNrows() - 1)) - (rows - 1 - i);
      if(newPadCol<0){
        newPadCol = 0;
      }
      padCol = newPadCol;
    }
    else{
      padCol = i ;
    }
    std::cout<<padCol<<std::endl;
  }
  //return modifiedWeightMatrix;
}

template <typename AFloat>
void ConvMultiply(TCpuMatrix<AFloat> &output,
                               const TCpuMatrix<AFloat> &input,
                               const TCpuMatrix<AFloat> &weights){
  size_t k = weights.GetNrows();
  size_t m = weights.GetNrows();
  size_t n = input.GetNcols();

  std::cout<<k<<" "<<m<<" "<<n<<std::endl;
  std::cout<<output.GetNrows()<<" "<<output.GetNcols()<<std::endl;
  std::cout<<input.GetNrows()<<" "<<input.GetNcols()<<std::endl;
  std::cout<<weights.GetNrows()<<" "<<weights.GetNcols()<<std::endl;

  for(size_t i = 0 ; i < m; i++){
    for(size_t j = 0 ; j < n; j++){
      output(i,j) = 0;
    }
  }

  for(size_t i = 0 ; i < m; i++){
    for(size_t j = 0 ; j < n; j++){
      for(size_t l = 0; l < k; l++){
        output(i,j) += weights(i,l)*input(l,j);
      }
    }
  }

}

template <typename AFloat>
void TCpu<AFloat>::GenerateColumnarMatrix(TCpuMatrix<AFloat> input,
                                    std::vector< TCpuMatrix<AFloat> > & inputColumnar){
  //TCpuMatrix<AFloat> inputColumnar(input.GetNrows()*input.GetNrows(),1);
  std::cout<<"To columnar "<<input.GetNrows()<<" "<<input.GetNcols()<<std::endl;
  for(size_t i = 0 ; i < input.GetNrows(); i++){
    for(size_t j = 0 ; j < input.GetNcols(); j++){
      std::cout<<"Input Indices "<<i<<" "<<j<<std::endl;
      std::cout<<input(i,j)<<std::endl;
      std::cout<<"Input Columnar Indices "<<(i*input.GetNcols())<<"+"<<j<<", "<<0<<std::endl;
      
      inputColumnar[0](i*input.GetNcols()+j,0) = input(i,j);
      
      std::cout<<"Input Columnar matrix : "<<inputColumnar[0](i*input.GetNcols()+j,0)<<std::endl;
    }
    std::cout<<std::endl;
  }
  //return inputColumnar;
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::TransConvLayerForward(std::vector<TCpuMatrix<AFloat>> & output,
                                    std::vector<TCpuMatrix<AFloat>> & derivatives,
                                    const std::vector<TCpuMatrix<AFloat>> &input,
                                    const TCpuMatrix<AFloat> &weights, const TCpuMatrix<AFloat> & biases,
                                    const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                    std::vector<TCpuMatrix<AFloat>> & /*  */)
{
  std::cout<<"Dimensions : "<<input.size()<<std::endl;
  for(size_t i = 0; i < output.size(); i++){

    TCpuMatrix<AFloat> outputTr(output[i].GetNcols()*output[i].GetNrows(),1);
    
    std::cout<<"Output Columnar Matrix "<<std::endl;
    for(size_t j = 0 ; j < outputTr.GetNrows(); j++){
      for(size_t k = 0; k < outputTr.GetNcols(); k++){
        std::cout<<outputTr(j,k)<<" ";
      }
      std::cout<<std::endl;
    }

    std::cout<<"Expected Output Matrix : "<<std::endl;
    std::cout<<output[i].GetNrows()<<" "<<output[i].GetNrows()<<std::endl;
    for(size_t j = 0 ; j < output[i].GetNrows(); j++){
      for(size_t k = 0; k < output[i].GetNcols(); k++){
        std::cout<<output[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<"Input Matrix : "<<std::endl;
    std::cout<<input[i].GetNrows()<<" "<<input[i].GetNrows()<<std::endl;
    for(size_t j = 0 ; j < input[i].GetNrows(); j++){
      for(size_t k = 0; k < input[i].GetNcols(); k++){
        std::cout<<input[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;

    TCpuMatrix<AFloat> inputTr(input[i].GetNrows()*input[i].GetNcols(),1);
    std::vector< TCpuMatrix<AFloat> > inputTrVector;
    inputTrVector.emplace_back(inputTr);
    GenerateColumnarMatrix(input[i],inputTrVector);
    
    std::cout<<"Generating Columnar Matrix"<<std::endl;
    std::cout<<"Input Tr Dimensions : "<<inputTr.GetNrows()<<" "<<inputTr.GetNcols()<<std::endl;
    std::cout<<"Input Transpose Matrix : "<<std::endl;
    for(size_t j = 0 ; j < inputTr.GetNrows(); j++){
      for(size_t k = 0; k < inputTr.GetNcols(); k++){
        std::cout<<inputTr(j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;

    std::cout<<"Generating Conv Matrix"<<std::endl;
    TCpuMatrix<AFloat> convMatrix(output[i].GetNcols(),inputTr.GetNrows());
    std::vector< TCpuMatrix<AFloat>> convMatrices;
    convMatrices.emplace_back(convMatrix);
    
    std::cout<<"Passed parameters : "<<output[i].GetNrows()<<" "<<output[i].GetNcols()<<" "<<inputTr.GetNrows()<<" "<<inputTr.GetNcols()<<std::endl;
    std::cout<<"Conv Matrix Dimensions : "<<convMatrices[i].GetNrows()<<" "<<convMatrices[i].GetNcols()<<std::endl;
    std::cout<<"Convolution Matrix of Weights "<<std::endl;
    
    GenerateConvMatrix(weights,convMatrices);
    
    for(size_t j = 0 ; j < convMatrices[i].GetNrows(); j++){
      for(size_t k = 0; k < convMatrices[i].GetNcols(); k++){
        std::cout<<convMatrices[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;

    std::cout<<"Multiplying  convMatrix and inputTr"<<std::endl;
    std::cout<<"Output "<<output[i].GetNrows()<<" "<<output[i].GetNcols()<<std::endl;
    std::cout<<"Weights "<<convMatrices[i].GetNrows()<<" "<<convMatrices[i].GetNcols()<<std::endl;
    std::cout<<"Input Tr "<<inputTr.GetNrows()<<" "<<inputTr.GetNcols()<<std::endl;

    Multiply(outputTr,convMatrices[i],inputTr);

    for(size_t j = 0 ; j < output[i].GetNrows(); j++){
      for(size_t k = 0; k < output[i].GetNcols(); k++){
        output[i](j,k) = outputTr(j*output[i].GetNcols()+k,0);
      }
    }

    std::cout<<"Output Transpose Matrix : "<<std::endl;
    for(size_t j = 0 ; j < outputTr.GetNrows(); j++){
      for(size_t k = 0; k < outputTr.GetNcols(); k++){
        std::cout<<outputTr(j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;

    std::cout<<"Output Matrix : "<<std::endl;
    for(size_t j = 0 ; j < output[i].GetNrows(); j++){
      for(size_t k = 0; k < output[i].GetNcols(); k++){
        std::cout<<output[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;    
    
    std::cout<<"Adding Biases"<<std::endl;
    AddConvBiases(output[i], biases);
    std::cout<<"Output Matrix : "<<std::endl;
    for(size_t j = 0 ; j < output[i].GetNrows(); j++){
      for(size_t k = 0; k < output[i].GetNcols(); k++){
        std::cout<<output[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
  }

   //evaluateDerivative<TCpu<AFloat>>(derivatives[i], activFunc, output[i]);
   //evaluate<TCpu<AFloat>>(output[i], activFunc);

   //TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(f, ROOT::TSeqI(input.size()));*/
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::TransConvLayerBackward(std::vector<TCpuMatrix<AFloat>> & output,
                                    std::vector<TCpuMatrix<AFloat>> & derivatives,
                                    const std::vector<TCpuMatrix<AFloat>> &input,
                                    const TCpuMatrix<AFloat> &weights, const TCpuMatrix<AFloat> & biases,
                                    const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                    std::vector<TCpuMatrix<AFloat>> & /*  */)
{
  std::cout<<"Dimensions : "<<input.size()<<std::endl;
  for(size_t i = 0; i < output.size(); i++){

    TCpuMatrix<AFloat> outputTr(output[i].GetNcols()*output[i].GetNrows(),1);
    
    std::cout<<"Output Columnar Matrix "<<std::endl;
    for(size_t j = 0 ; j < outputTr.GetNrows(); j++){
      for(size_t k = 0; k < outputTr.GetNcols(); k++){
        std::cout<<outputTr(j,k)<<" ";
      }
      std::cout<<std::endl;
    }

    std::cout<<"Expected Output Matrix : "<<std::endl;
    std::cout<<output[i].GetNrows()<<" "<<output[i].GetNrows()<<std::endl;
    for(size_t j = 0 ; j < output[i].GetNrows(); j++){
      for(size_t k = 0; k < output[i].GetNcols(); k++){
        std::cout<<output[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<"Input Matrix : "<<std::endl;
    std::cout<<input[i].GetNrows()<<" "<<input[i].GetNrows()<<std::endl;
    for(size_t j = 0 ; j < input[i].GetNrows(); j++){
      for(size_t k = 0; k < input[i].GetNcols(); k++){
        std::cout<<input[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;

    TCpuMatrix<AFloat> inputTr(input[i].GetNrows()*input[i].GetNcols(),1);
    std::vector< TCpuMatrix<AFloat> > inputTrVector;
    inputTrVector.emplace_back(inputTr);
    GenerateColumnarMatrix(input[i],inputTrVector);
    
    std::cout<<"Generating Columnar Matrix"<<std::endl;
    std::cout<<"Input Tr Dimensions : "<<inputTr.GetNrows()<<" "<<inputTr.GetNcols()<<std::endl;
    std::cout<<"Input Transpose Matrix : "<<std::endl;
    for(size_t j = 0 ; j < inputTr.GetNrows(); j++){
      for(size_t k = 0; k < inputTr.GetNcols(); k++){
        std::cout<<inputTr(j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;

    std::cout<<"Generating Conv Matrix"<<std::endl;
    TCpuMatrix<AFloat> convMatrix(output[i].GetNcols(),inputTr.GetNrows());
    std::vector< TCpuMatrix<AFloat>> convMatrices;
    convMatrices.emplace_back(convMatrix);
    
    std::cout<<"Passed parameters : "<<output[i].GetNrows()<<" "<<output[i].GetNcols()<<" "<<inputTr.GetNrows()<<" "<<inputTr.GetNcols()<<std::endl;
    std::cout<<"Conv Matrix Dimensions : "<<convMatrices[i].GetNrows()<<" "<<convMatrices[i].GetNcols()<<std::endl;
    std::cout<<"Convolution Matrix of Weights "<<std::endl;
    
    GenerateTransConvMatrix(weights,convMatrices);
    
    for(size_t j = 0 ; j < convMatrices[i].GetNrows(); j++){
      for(size_t k = 0; k < convMatrices[i].GetNcols(); k++){
        std::cout<<convMatrices[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;

    std::cout<<"Multiplying  convMatrix and inputTr"<<std::endl;
    std::cout<<"Output "<<output[i].GetNrows()<<" "<<output[i].GetNcols()<<std::endl;
    std::cout<<"Weights "<<convMatrices[i].GetNrows()<<" "<<convMatrices[i].GetNcols()<<std::endl;
    std::cout<<"Input Tr "<<inputTr.GetNrows()<<" "<<inputTr.GetNcols()<<std::endl;

    Multiply(outputTr,convMatrices[i],inputTr);

    for(size_t j = 0 ; j < output[i].GetNrows(); j++){
      for(size_t k = 0; k < output[i].GetNcols(); k++){
        output[i](j,k) = outputTr(j*output[i].GetNcols()+k,0);
      }
    }

    std::cout<<"Output Transpose Matrix : "<<std::endl;
    for(size_t j = 0 ; j < outputTr.GetNrows(); j++){
      for(size_t k = 0; k < outputTr.GetNcols(); k++){
        std::cout<<outputTr(j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;

    std::cout<<"Output Matrix : "<<std::endl;
    for(size_t j = 0 ; j < output[i].GetNrows(); j++){
      for(size_t k = 0; k < output[i].GetNcols(); k++){
        std::cout<<output[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;    
    
    std::cout<<"Adding Biases"<<std::endl;
    AddConvBiases(output[i], biases);
    std::cout<<"Output Matrix : "<<std::endl;
    for(size_t j = 0 ; j < output[i].GetNrows(); j++){
      for(size_t k = 0; k < output[i].GetNcols(); k++){
        std::cout<<output[i](j,k)<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
  }

   //evaluateDerivative<TCpu<AFloat>>(derivatives[i], activFunc, output[i]);
   //evaluate<TCpu<AFloat>>(output[i], activFunc);

   //TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(f, ROOT::TSeqI(input.size()));*/
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::ConvLayerBackward(std::vector<TCpuMatrix<AFloat>> &activationGradientsBackward,
                                     TCpuMatrix<AFloat> &weightGradients, TCpuMatrix<AFloat> &biasGradients,
                                     std::vector<TCpuMatrix<AFloat>> &df,
                                     const std::vector<TCpuMatrix<AFloat>> &activationGradients,
                                     const TCpuMatrix<AFloat> &weights,
                                     const std::vector<TCpuMatrix<AFloat>> &activationsBackward, size_t batchSize,
                                     size_t inputHeight, size_t inputWidth, size_t depth, size_t height, size_t width,
                                     size_t filterDepth, size_t filterHeight, size_t filterWidth, size_t nLocalViews)
{
   // Update derivatives
   //    size_t m, n;
   //    m = activationGradients[0].GetNrows();
   //    n = activationGradients[0].GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      // Compute element-wise product.
      Hadamard(df[i], activationGradients[i]);
   }

   // Calculate the activation gradients of the previous layer
   CalculateConvActivationGradients(activationGradientsBackward, df, weights, batchSize, inputHeight, inputWidth, depth,
                                    height, width, filterDepth, filterHeight, filterWidth);

   // Calculate the weight gradients
   CalculateConvWeightGradients(weightGradients, df, activationsBackward, batchSize, inputHeight, inputWidth, depth,
                                height, width, filterDepth, filterHeight, filterWidth, nLocalViews);

   // Calculate the bias gradients
   CalculateConvBiasGradients(biasGradients, df, batchSize, depth, nLocalViews);
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvActivationGradients(std::vector<TCpuMatrix<AFloat>> &activationGradientsBackward,
                                                    const std::vector<TCpuMatrix<AFloat>> &df,
                                                    const TCpuMatrix<AFloat> &weights, size_t batchSize,
                                                    size_t inputHeight, size_t inputWidth, size_t depth, size_t height,
                                                    size_t width, size_t filterDepth, size_t filterHeight,
                                                    size_t filterWidth)
{
   if (activationGradientsBackward.size() == 0) return;

   for (size_t i = 0; i < activationGradientsBackward.size(); i++) {
      activationGradientsBackward[i].Zero();
   }

   // Transform the weights

   //TMVA_DNN_PrintTCpuMatrix(weights,"weights");
   // filter depth must be same as input depth
   TCpuMatrix<AFloat> rotWeights(filterDepth, depth * filterHeight * filterWidth);
   RotateWeights(rotWeights, weights, filterDepth, filterHeight, filterWidth, weights.GetNrows());
   //TMVA_DNN_PrintTCpuMatrix(rotWeights,"rot-weights");

   // Calculate the zero paddings
   size_t tempZeroPaddingHeight = (size_t)(floor((inputHeight - height + filterHeight - 1) / 2));
   size_t tempZeroPaddingWidth = (size_t)(floor((inputWidth - width + filterWidth - 1) / 2));

   // size_t tempZeroPaddingHeight = 1;
   // size_t tempZeroPaddingWidth = 1;
   
   // Calculate the number of local views and the number of pixles in each view
   size_t tempNLocalViews = inputHeight * inputWidth;
   size_t tempNLocalViewPixels = depth * filterHeight * filterWidth;

   size_t tempStrideRows = 1;
   size_t tempStrideCols = 1;

   // An entire convolution follows

    std::vector<int> vIndices( tempNLocalViews * tempNLocalViewPixels );
    Im2colIndices(vIndices, df[0], tempNLocalViews, height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
             tempZeroPaddingHeight, tempZeroPaddingWidth);


    //for (size_t i = 0; i < batchSize; i++) {
    R__ASSERT(batchSize == df.size() );
    R__ASSERT(batchSize == activationGradientsBackward.size() );
    auto f = [&] (UInt_t i)
   {
   
       // Im2col(dfTr, df[i], height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
       //       tempZeroPaddingHeight, tempZeroPaddingWidth);

      TCpuMatrix<AFloat> dfTr(tempNLocalViews, tempNLocalViewPixels);
      
      Im2colFast(dfTr, df[i], vIndices); 

       //TMVA_DNN_PrintTCpuMatrix(df[i],"df[i]");
       //TMVA_DNN_PrintTCpuMatrix(dfTr,"dfTr");

       MultiplyTranspose(activationGradientsBackward[i], rotWeights, dfTr);

       //TMVA_DNN_PrintTCpuMatrix(activationGradientsBackward[i],"activGrad-result");

   };

    TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(f, ROOT::TSeqI( batchSize ) );
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvWeightGradients(TCpuMatrix<AFloat> &weightGradients,
                                                const std::vector<TCpuMatrix<AFloat>> &df,
                                                const std::vector<TCpuMatrix<AFloat>> &activationsBackward,
                                                size_t batchSize, size_t inputHeight, size_t inputWidth, size_t depth,
                                                size_t height, size_t width, size_t filterDepth, size_t filterHeight,
                                                size_t filterWidth, size_t nLocalViews)
{
   // reinitialize the weight gradients to 0
   weightGradients.Zero();

   const size_t filterSize = filterHeight * filterWidth;
   const size_t nLocalViewPixels = filterDepth * filterHeight * filterWidth;
   R__ASSERT( weightGradients.GetNcols() == filterDepth * filterHeight * filterWidth);

   const size_t tempStrideRows = 1;
   const size_t tempStrideCols = 1;
      
      // Calculate the zero paddings from the input height and width (assume stride =1 )      
   const size_t tempZeroPaddingHeight = (height - inputHeight + filterHeight - 1) / 2;
   const size_t tempZeroPaddingWidth = (width - inputWidth + filterWidth - 1) / 2;


   // convolution
   
   

   std::vector<int> vIndices(nLocalViews * nLocalViewPixels );
   Im2colIndices(vIndices, activationsBackward[0], nLocalViews, inputHeight, inputWidth, filterHeight , filterWidth,
             tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);
   
   //std::cout << "do back-propagation in conv layer - compute weight gradient" << std::endl;

   std::vector< TCpuMatrix<AFloat> > vres;//(batchSize); 
   for (size_t i = 0; i < batchSize; i++) {
      vres.emplace_back(depth, nLocalViewPixels);
      //TMVA_DNN_PrintTCpuMatrix(df[i],"df");
      //TMVA_DNN_PrintTCpuMatrix(activationsBackward[i],"df");
      
   }
   
   auto fmap = [&](int i) { 
 
      //TMVA_DNN_PrintTCpuMatrix(df[i],"df-i");
      TCpuMatrix<AFloat> xTr(nLocalViews, nLocalViewPixels);
      TCpuMatrix<AFloat> res(depth, nLocalViewPixels);

      //computing t he gradient is equivalent of doing a convolution of the input using as conv kernel the delta's (the df[] values) 
      //N.B. only stride values=1 are now supported
 
      //xTr.Zero(); 
      // Im2col(xTr, const_cast<TCpuMatrix<AFloat> &>(activationsBackward[i]), inputHeight, inputWidth, filterHeight , filterWidth,
      //        tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);
      Im2colFast(xTr, activationsBackward[i], vIndices);

      //std::cout << "doing im2colfast" << std::endl;
      //TMVA_DNN_PrintTCpuMatrix(xTr,"xTr-i");
      //TMVA_DNN_PrintTCpuMatrix(activationsBackward[i],"actbackward-i");
      Multiply(vres[i], df[i], xTr);
      //TMVA_DNN_PrintTCpuMatrix(vres[i],"res_ofMT");

      return;
      //return res;
   };

   TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(fmap, ROOT::TSeqI( batchSize ) );

//   auto freduce = [&](const std::vector<TCpuMatrix<AFloat>> & vres) { 
      R__ASSERT(vres.size() == batchSize); 
      for (size_t i = 0; i < batchSize; i++) {
         //TMVA_DNN_PrintTCpuMatrix(vres[i],"res");
         for (size_t j = 0; j < depth; j++) {
            for (size_t k = 0; k < filterDepth; k++) {
               size_t kOffset = k * filterSize; 
               for (size_t l = 0; l < filterSize; l++) {
                  //weightGradients(j, k * (filterHeight * filterWidth) + l) += res(k, (tempNLocalViews - 1) - l);
                  weightGradients(j, kOffset + l) += vres[i](j,  kOffset + l);
               }
            }
         }
         // TMVA_DNN_PrintTCpuMatrix(weightGradients,"weights_i");
      }
      //  };
  
   //TCpuMatrix<AFloat>::GetThreadExecutor().MapReduce(fmap, ROOT::TSeqI( batchSize ) , freduce);
   //TMVA_DNN_PrintTCpuMatrix(weightGradients,"W-Grad");
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvBiasGradients(TCpuMatrix<AFloat> &biasGradients, const std::vector<TCpuMatrix<AFloat>> &df,
                                              size_t batchSize, size_t depth, size_t nLocalViews)
{
   biasGradients.Zero();
   for (size_t i = 0; i < depth; i++) {
      AFloat sum = 0;
      for (size_t j = 0; j < nLocalViews; j++) {
         for (size_t k = 0; k < batchSize; k++) {
            sum += df[k](i, j);
         }
      }
      biasGradients(i, 0) = sum;
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Downsample(TCpuMatrix<AFloat> &A, TCpuMatrix<AFloat> &B, const TCpuMatrix<AFloat> &C,
                              size_t imgHeight, size_t imgWidth, size_t fltHeight, size_t fltWidth, size_t strideRows,
                              size_t strideCols)
{
   // image boudaries
   int imgHeightBound = imgHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;

   // centers
   for (int i = fltHeight / 2; i <= imgHeightBound; i += strideRows) {
      for (int j = fltWidth / 2; j <= imgWidthBound; j += strideCols) {
         // within local views
         for (int m = 0; m < (Int_t)C.GetNrows(); m++) {
            AFloat value = -std::numeric_limits<AFloat>::max();

            for (int k = i - fltHeight / 2; k <= Int_t(i + (fltHeight - 1) / 2); k++) {
               for (int l = j - fltWidth / 2; l <= Int_t(j + (fltWidth - 1) / 2); l++) {
                  if (C(m, k * imgWidth + l) > value) {
                     value = C(m, k * imgWidth + l);
                     B(m, currLocalView) = k * imgWidth + l;
                  }
               }
            }
            A(m, currLocalView) = value;
         }
         currLocalView++;
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::MaxPoolLayerBackward(TCpuMatrix<AFloat> &activationGradientsBackward,
                                        const TCpuMatrix<AFloat> &activationGradients,
                                        const TCpuMatrix<AFloat> &indexMatrix,
                                        size_t /* imgHeight */,
                                        size_t /* imgWidth */,
                                        size_t /* fltHeight */,
                                        size_t /* fltWidth */,
                                        size_t /* strideRows */,
                                        size_t /* strideCols */,
                                        size_t nLocalViews)
{
   size_t depth = activationGradientsBackward.GetNrows();

   for (size_t j = 0; j < depth; j++) {
      // initialize to zeros
      for (size_t t = 0; t < (size_t)activationGradientsBackward.GetNcols(); t++) {
         activationGradientsBackward(j, t) = 0;
      }

      // set values
      for (size_t k = 0; k < nLocalViews; k++) {
         AFloat grad = activationGradients(j, k);
         size_t winningIdx = indexMatrix(j, k);
         activationGradientsBackward(j, winningIdx) += grad;
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Reshape(TCpuMatrix<AFloat> &A, const TCpuMatrix<AFloat> &B)
{
   size_t nColsA = A.GetNcols();
   size_t nColsB = B.GetNcols();

   for (size_t i = 0; i < A.GetNrows(); i++) {
      for (size_t j = 0; j < A.GetNcols(); j++) {
         size_t nElem = i * nColsA + j;
         A(i, j) = B(nElem / nColsB, nElem % nColsB);
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Flatten(TCpuMatrix<AFloat> &A, const std::vector<TCpuMatrix<AFloat>> &B, size_t size, size_t nRows,
                           size_t nCols)
{
   for (size_t i = 0; i < (size_t)size; i++) {
      for (size_t j = 0; j < (size_t)nRows; j++) {
         for (size_t k = 0; k < (size_t)nCols; k++) {
            A(i, j * nCols + k) = B[i](j, k);
         }
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Deflatten(std::vector<TCpuMatrix<AFloat>> &A, const TCpuMatrix<AFloat> &B, size_t size, size_t nRows,
                             size_t nCols)
{
   for (size_t i = 0; i < (size_t)size; i++) {
      for (size_t j = 0; j < (size_t)nRows; j++) {
         for (size_t k = 0; k < (size_t)nCols; k++) {
            A[i](j, k) = B(i, j * nCols + k);
         }
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TCpu<AReal>::Rearrange(std::vector<TCpuMatrix<AReal>> &out, const std::vector<TCpuMatrix<AReal>> &in)
{
   // B x T x D out --- T x B x D in*/
   size_t B = out.size();
   size_t T = out[0].GetNrows();
   size_t D = out[0].GetNcols();
   if ((T != in.size()) || (B != in[0].GetNrows()) || (D != in[0].GetNcols())) {
      std::cout << "Incompatible Dimensions\n"
                << in.size() << "x" << in[0].GetNrows() << "x" << in[0].GetNcols() << " --> " << B << "x" << T << "x"
                << D << "\n";
      return;
   }
   for (size_t i = 0; i < B; ++i) {
      for (size_t j = 0; j < T; ++j) {
         for (size_t k = 0; k < D; ++k) {
            out[i](j, k) = in[j](i, k);
         }
      }
   }
   return;
}

} // namespace DNN
} // namespace TMVA