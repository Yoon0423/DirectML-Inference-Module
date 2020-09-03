// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "MaxPoolOperator.hpp"
#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"
#include "DirectMLHelperFunction.hpp"

namespace colombia_supremo {

MaxPoolOperator::MaxPoolOperator(TensorShape inputShape,
                                 TensorShape outputShape, const uint32_t stride,
                                 const uint32_t padding,
                                 const uint32_t kernelSize)
    : Operator(std::move(inputShape), std::move(outputShape)), mStride(stride),
      mPadding(padding) {
  // create inputTensorDesc
  DML_BUFFER_TENSOR_DESC inputBufferTensorDesc;
  TensorSizes inputStrides =
      dml_helper::CalculateStrides(mInputTensor->getShapeRef());
  {
    inputBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    inputBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    inputBufferTensorDesc.DimensionCount = mInputTensor->getShapeRef().size();
    inputBufferTensorDesc.Sizes = mInputTensor->getShapeRef().data();
    inputBufferTensorDesc.Strides = inputStrides.data();
    inputBufferTensorDesc.GuaranteedBaseOffsetAlignment = 0;
    inputBufferTensorDesc.TotalTensorSizeInBytes =
        mInputTensor->getTensorBufferSize();
  }

  DML_TENSOR_DESC inputTensorDesc;
  inputTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
  inputTensorDesc.Desc = &inputBufferTensorDesc;

  // create outputTensorDesc
  DML_BUFFER_TENSOR_DESC outputBufferTensorDesc;
  TensorSizes outputStrides =
      dml_helper::CalculateStrides(mOutputTensor->getShapeRef());
  {
    outputBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    outputBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    outputBufferTensorDesc.DimensionCount = mOutputTensor->getShapeRef().size();
    outputBufferTensorDesc.Sizes = mOutputTensor->getShapeRef().data();
    outputBufferTensorDesc.Strides = outputStrides.data();
    outputBufferTensorDesc.GuaranteedBaseOffsetAlignment = 0;
    outputBufferTensorDesc.TotalTensorSizeInBytes =
        mOutputTensor->getTensorBufferSize();
  }

  DML_TENSOR_DESC outputTensorDesc{};
  outputTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
  outputTensorDesc.Desc = &outputBufferTensorDesc;

  // create max pooling operator descriptor
  DML_MAX_POOLING_OPERATOR_DESC maxPoolOperatorDesc;
  uint32_t strides[] = {mStride, mStride};
  uint32_t startPaddings[] = {mPadding, mPadding};
  uint32_t endPaddings[] = {mPadding, mPadding};
  uint32_t kernelShape[] = {kernelSize, kernelSize};
  {
    maxPoolOperatorDesc.InputTensor = &inputTensorDesc;
    maxPoolOperatorDesc.OutputTensor = &outputTensorDesc;
    maxPoolOperatorDesc.DimensionCount = 2;
    maxPoolOperatorDesc.Strides = strides;
    maxPoolOperatorDesc.WindowSize = kernelShape;
    maxPoolOperatorDesc.StartPadding = startPaddings;
    maxPoolOperatorDesc.EndPadding = endPaddings;
  }

  DML_OPERATOR_DESC operatorDesc{};
  operatorDesc.Type = DML_OPERATOR_MAX_POOLING;
  operatorDesc.Desc = &maxPoolOperatorDesc;

  DeviceManager &deviceManager = DeviceManager::getInstance();

  winrt::com_ptr<IDMLOperator> dmlOperator;
  winrt::check_hresult(deviceManager.mDMLDevice->CreateOperator(
      &operatorDesc, __uuidof(dmlOperator), dmlOperator.put_void()));

  winrt::check_hresult(deviceManager.mDMLDevice->CompileOperator(
      dmlOperator.get(), DML_EXECUTION_FLAG_NONE, __uuidof(mCompiledOperator),
      mCompiledOperator.put_void()));

  InitBindingTables();
}

void MaxPoolOperator::InitBindingTables() {
  auto defaultDevice = DeviceManager::getInstance().GetDefault();

  winrt::com_ptr<IDMLOperatorInitializer> operatorInitializer;
  IDMLCompiledOperator *dmlCompiledOperators[] = {mCompiledOperator.get()};
  winrt::check_hresult(defaultDevice->mDMLDevice->CreateOperatorInitializer(
      1, dmlCompiledOperators, __uuidof(operatorInitializer),
      operatorInitializer.put_void()));

  uint32_t descriptorCount =
      operatorInitializer->GetBindingProperties().RequiredDescriptorCount;

  if (descriptorCount <
      mCompiledOperator->GetBindingProperties().RequiredDescriptorCount) {
    descriptorCount =
        mCompiledOperator->GetBindingProperties().RequiredDescriptorCount;
  }

  {
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    winrt::check_hresult(defaultDevice->mD3D12Device->CreateDescriptorHeap(
        &descriptorHeapDesc, _uuidof(mExecDescriptorHeap),
        mExecDescriptorHeap.put_void()));
  }

  ID3D12DescriptorHeap *descriptorHeaps[] = {mExecDescriptorHeap.get()};
  defaultDevice->mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                                  descriptorHeaps);

  {
    DML_BINDING_TABLE_DESC tableDesc = {
        operatorInitializer.get(),
        mExecDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
        mExecDescriptorHeap->GetGPUDescriptorHandleForHeapStart(),
        descriptorCount};

    winrt::check_hresult(defaultDevice->mDMLDevice->CreateBindingTable(
        &tableDesc, __uuidof(mExecBindingTable), mExecBindingTable.put_void()));
  }

  defaultDevice->mCommandRecorder->RecordDispatch(
      defaultDevice->mCommandList.get(), operatorInitializer.get(),
      mExecBindingTable.get());

  defaultDevice->CloseExecuteResetWait();

  {
    ID3D12DescriptorHeap *descriptorHeaps[] = {mExecDescriptorHeap.get()};
    defaultDevice->mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                                    descriptorHeaps);

    DML_BINDING_TABLE_DESC tableDesc = {
        mCompiledOperator.get(),
        mExecDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
        mExecDescriptorHeap->GetGPUDescriptorHandleForHeapStart(),
        descriptorCount};

    winrt::check_hresult(mExecBindingTable->Reset(&tableDesc));
  }

  DML_BUFFER_BINDING outputBufferBinding{mOutputTensor->getBufferPtr(), 0,
                                         mOutputTensor->getTensorBufferSize()};
  DML_BINDING_DESC outputBinding{DML_BINDING_TYPE_BUFFER, &outputBufferBinding};

  DML_BUFFER_BINDING inputBufferBinding{mInputTensor->getBufferPtr(), 0,
                                        mInputTensor->getTensorBufferSize()};
  DML_BINDING_DESC inputBinding{DML_BINDING_TYPE_BUFFER, &inputBufferBinding};

  DML_BINDING_DESC inputBindings[] = {inputBinding};
  mExecBindingTable->BindInputs(1, inputBindings);
  mExecBindingTable->BindOutputs(1, &outputBinding);

  defaultDevice->mCommandRecorder->RecordDispatch(
      defaultDevice->mCommandList.get(), mCompiledOperator.get(),
      mExecBindingTable.get());

  defaultDevice->CloseExecuteResetWait();
}

} // namespace colombia_supremo
