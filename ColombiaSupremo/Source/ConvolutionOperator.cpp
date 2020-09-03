// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "ConvolutionOperator.hpp"
#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"
#include "DirectMLHelperFunction.hpp"

namespace colombia_supremo {

ConvolutionOperator::ConvolutionOperator(
    TensorShape inputShape, TensorShape outputShape,
    const std::shared_ptr<WeightTensor> weightTensor,
    const std::shared_ptr<WeightTensor> biasTensor, const uint32_t stride,
    const uint32_t padding, const bool isReluActivated)
    : Operator(inputShape, outputShape), mWeightTensor(weightTensor),
      mBiasTensor(biasTensor), mStride(stride), mPadding(padding) {
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

  // create filterTensorDes
  DML_BUFFER_TENSOR_DESC filterBufferTensorDesc;
  TensorSizes filterStrides =
      dml_helper::CalculateStrides(mWeightTensor->getShapeRef());
  {
    filterBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    filterBufferTensorDesc.Flags = DML_TENSOR_FLAG_OWNED_BY_DML;
    filterBufferTensorDesc.DimensionCount = mWeightTensor->getShapeRef().size();
    filterBufferTensorDesc.Sizes = mWeightTensor->getShapeRef().data();
    filterBufferTensorDesc.Strides = filterStrides.data();
    filterBufferTensorDesc.GuaranteedBaseOffsetAlignment = 0;
    filterBufferTensorDesc.TotalTensorSizeInBytes =
        mWeightTensor->getTensorBufferSize();
  }

  DML_TENSOR_DESC filterTensorDesc{};
  filterTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
  filterTensorDesc.Desc = &filterBufferTensorDesc;

  // create biasTensorDesc
  DML_BUFFER_TENSOR_DESC biasBufferTensorDesc;
  TensorSizes biasStrides;
  DML_TENSOR_DESC biasTensorDesc{};
  if (mBiasTensor != nullptr) {
    biasStrides = dml_helper::CalculateStrides(mBiasTensor->getShapeRef());
    {
      biasBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
      biasBufferTensorDesc.Flags = DML_TENSOR_FLAG_OWNED_BY_DML;
      biasBufferTensorDesc.DimensionCount = mBiasTensor->getShapeRef().size();
      biasBufferTensorDesc.Sizes = mBiasTensor->getShapeRef().data();
      biasBufferTensorDesc.Strides = biasStrides.data();
      biasBufferTensorDesc.GuaranteedBaseOffsetAlignment = 0;
      biasBufferTensorDesc.TotalTensorSizeInBytes =
          mBiasTensor->getTensorBufferSize();
    }

    biasTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
    biasTensorDesc.Desc = &biasBufferTensorDesc;
  }

  // create activation operator desc
  DML_ACTIVATION_RELU_OPERATOR_DESC reluDesc;
  {
    reluDesc.InputTensor = nullptr;
    reluDesc.OutputTensor = nullptr;
  }

  DML_OPERATOR_DESC activationDesc{};
  activationDesc.Type = DML_OPERATOR_ACTIVATION_RELU;
  activationDesc.Desc = &reluDesc;

  // create convolution operator descriptor
  DML_CONVOLUTION_OPERATOR_DESC convolutionOperatorDesc;
  uint32_t strides[] = {mStride, mStride};
  uint32_t dilations[] = {1, 1};
  uint32_t StartPaddings[] = {mPadding, mPadding};
  uint32_t endPaddings[] = {mPadding, mPadding};
  uint32_t outputPaddings[] = {0, 0};

  {
    convolutionOperatorDesc.InputTensor = &inputTensorDesc;
    convolutionOperatorDesc.FilterTensor = &filterTensorDesc;
    convolutionOperatorDesc.BiasTensor =
        mBiasTensor != nullptr ? &biasTensorDesc : nullptr;
    convolutionOperatorDesc.OutputTensor = &outputTensorDesc;
    convolutionOperatorDesc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    convolutionOperatorDesc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
    convolutionOperatorDesc.DimensionCount = 2;
    convolutionOperatorDesc.FusedActivation =
        isReluActivated == true ? &activationDesc : nullptr;
    convolutionOperatorDesc.Strides = strides;
    convolutionOperatorDesc.Dilations = dilations;
    convolutionOperatorDesc.StartPadding = StartPaddings;
    convolutionOperatorDesc.EndPadding = endPaddings;
    convolutionOperatorDesc.OutputPadding = outputPaddings;
    convolutionOperatorDesc.GroupCount = 1;
  }

  DML_OPERATOR_DESC operatorDesc{};
  operatorDesc.Type = DML_OPERATOR_CONVOLUTION;
  operatorDesc.Desc = &convolutionOperatorDesc;

  DeviceManager &deviceManager = DeviceManager::getInstance();

  winrt::com_ptr<IDMLOperator> dmlOperator;
  winrt::check_hresult(deviceManager.mDMLDevice->CreateOperator(
      &operatorDesc, __uuidof(dmlOperator), dmlOperator.put_void()));

  winrt::check_hresult(deviceManager.mDMLDevice->CompileOperator(
      dmlOperator.get(), DML_EXECUTION_FLAG_NONE, __uuidof(mCompiledOperator),
      mCompiledOperator.put_void()));

  InitBindingTables();
}

std::shared_ptr<WeightTensor> ConvolutionOperator::getWeightTensor() {
  return mWeightTensor;
}

void ConvolutionOperator::InitBindingTables() {
  auto defaultDevice = DeviceManager::getInstance().GetDefault();

  winrt::com_ptr<IDMLOperatorInitializer> operatorInitializer;
  IDMLCompiledOperator *dmlCompiledOperators[] = {mCompiledOperator.get()};
  winrt::check_hresult(defaultDevice->mDMLDevice->CreateOperatorInitializer(
      1, dmlCompiledOperators, __uuidof(operatorInitializer),
      operatorInitializer.put_void()));

  {
    uint32_t descriptorCount =
        operatorInitializer->GetBindingProperties().RequiredDescriptorCount;

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    winrt::check_hresult(defaultDevice->mD3D12Device->CreateDescriptorHeap(
        &descriptorHeapDesc, _uuidof(mInitDescriptorHeap),
        mInitDescriptorHeap.put_void()));
  }

  {
    uint32_t descriptorCount =
        mCompiledOperator->GetBindingProperties().RequiredDescriptorCount;

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    winrt::check_hresult(defaultDevice->mD3D12Device->CreateDescriptorHeap(
        &descriptorHeapDesc, _uuidof(mExecDescriptorHeap),
        mExecDescriptorHeap.put_void()));
  }

  {
    auto bindingProp = mCompiledOperator->GetBindingProperties();

    if (bindingProp.PersistentResourceSize > 0) {
      D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
          bindingProp.PersistentResourceSize,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

      winrt::check_hresult(defaultDevice->mD3D12Device->CreateCommittedResource(
          &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
          D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_COMMON,
          nullptr, __uuidof(mPersistentResource),
          mPersistentResource.put_void()));
    }

    if (bindingProp.TemporaryResourceSize > 0) {
      assert("ConvolutionOperatorTmp cannot handle temporary resource");
    }
  }

  {
    auto bindingProps = operatorInitializer->GetBindingProperties();
    assert(bindingProps.PersistentResourceSize == 0);

    ID3D12DescriptorHeap *descriptorHeaps[] = {mInitDescriptorHeap.get()};
    defaultDevice->mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                                    descriptorHeaps);

    DML_BINDING_TABLE_DESC tableDesc = {
        operatorInitializer.get(),
        mInitDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
        mInitDescriptorHeap->GetGPUDescriptorHandleForHeapStart(),
        bindingProps.RequiredDescriptorCount};

    winrt::check_hresult(defaultDevice->mDMLDevice->CreateBindingTable(
        &tableDesc, __uuidof(mInitBindingTable), mInitBindingTable.put_void()));

    if (bindingProps.TemporaryResourceSize > 0) {
      assert("ConvolutionOperatorTmp initBindingTable cannot handle temporary "
             "resource size");
    }
  }

  // TODO: should be global constant
  const DML_BUFFER_BINDING emptyBufferBinding = {nullptr, 0, 0};
  const DML_BINDING_DESC emptyBindingDesc = {DML_BINDING_TYPE_NONE, nullptr};

  DML_BUFFER_BINDING convBufferBindings[3] = {
      emptyBufferBinding,
      {mWeightTensor->getBufferPtr(), 0,
       mWeightTensor->getBufferPtr()->GetDesc().Width},
      {mBiasTensor->getBufferPtr(), 0,
       mBiasTensor->getBufferPtr()->GetDesc().Width}};

  DML_BUFFER_ARRAY_BINDING convBufferArrayBindings = {3, convBufferBindings};
  DML_BINDING_DESC convInBindings = {DML_BINDING_TYPE_BUFFER_ARRAY,
                                     &convBufferArrayBindings};

  mInitBindingTable->BindInputs(1, &convInBindings);

  DML_BUFFER_BINDING convPersistentBuffer;
  DML_BINDING_DESC convPersistentBinding;
  {
    if (mPersistentResource.get() != nullptr) {
      convPersistentBuffer = {mPersistentResource.get(), 0,
                              mPersistentResource->GetDesc().Width};
      convPersistentBinding = {DML_BINDING_TYPE_BUFFER, &convPersistentBuffer};
    } else {
      convPersistentBinding = emptyBindingDesc;
    }

    mInitBindingTable->BindOutputs(1, &convPersistentBinding);
  }

  defaultDevice->mCommandRecorder->RecordDispatch(
      defaultDevice->mCommandList.get(), operatorInitializer.get(),
      mInitBindingTable.get());

  {
    auto bindingProps = mCompiledOperator->GetBindingProperties();

    ID3D12DescriptorHeap *descriptorHeaps[] = {mExecDescriptorHeap.get()};
    defaultDevice->mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                                    descriptorHeaps);

    DML_BINDING_TABLE_DESC tableDesc = {
        mCompiledOperator.get(),
        mExecDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
        mExecDescriptorHeap->GetGPUDescriptorHandleForHeapStart(),
        bindingProps.RequiredDescriptorCount};

    winrt::check_hresult(defaultDevice->mDMLDevice->CreateBindingTable(
        &tableDesc, __uuidof(mExecBindingTable), mExecBindingTable.put_void()));
  }

  defaultDevice->CloseExecuteResetWait();

  // can bind only once for exec binding table?
  DML_BUFFER_BINDING outputBufferBinding{mOutputTensor->getBufferPtr(), 0,
                                         mOutputTensor->getTensorBufferSize()};
  DML_BINDING_DESC outputBinding{DML_BINDING_TYPE_BUFFER, &outputBufferBinding};

  DML_BUFFER_BINDING inputBufferBinding{mInputTensor->getBufferPtr(), 0,
                                        mInputTensor->getTensorBufferSize()};
  DML_BINDING_DESC inputBinding{DML_BINDING_TYPE_BUFFER, &inputBufferBinding};

  DML_BINDING_DESC inputBindings[] = {inputBinding, emptyBindingDesc,
                                      emptyBindingDesc};
  mExecBindingTable->BindInputs(3, inputBindings);
  mExecBindingTable->BindOutputs(1, &outputBinding);

  if (mPersistentResource.get() != nullptr) {
    mExecBindingTable->BindPersistentResource(&convPersistentBinding);
  }
}

} // namespace colombia_supremo
