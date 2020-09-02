// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "FullyConnectionOperator.hpp"
#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"
#include "DirectMLHelperFunction.hpp"

namespace colombia_supremo {

FullyConnectionOperator::FullyConnectionOperator(
    const uint32_t inputLength, const uint32_t outputLength,
    const std::shared_ptr<WeightTensor> weightTensor,
    const std::shared_ptr<TensorRawData> biasData)
    : Operator(std::move(TensorShape({1, 1, 2, inputLength})),
               std::move(TensorShape({1, 1, 2, outputLength}))),
      mWeightTensor(weightTensor) {
  // create mBiasTensor
  if (biasData != nullptr) {
    TensorShape shape({1, 1, 2, outputLength});
    const size_t size = static_cast<size_t>(2 * outputLength);
    TensorRawData data;
    data.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      data.emplace_back(0.f);
    }

    for (size_t i = 0; i < biasData->size(); ++i) {
      data[i] = biasData->data()[i];
    }

    mBiasTensor = std::make_shared<WeightTensor>(data, std::move(shape));
  }

  // create aTensorDesc
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

  // create bTensorDesc
  DML_BUFFER_TENSOR_DESC weightBufferTensorDesc;
  TensorSizes weightStrides =
      dml_helper::CalculateStrides(mWeightTensor->getShapeRef());
  {
    // DML_TENSOR_FLAG_OWNED_BY_DML
    weightBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    weightBufferTensorDesc.Flags = DML_TENSOR_FLAG_OWNED_BY_DML;
    weightBufferTensorDesc.DimensionCount = mWeightTensor->getShapeRef().size();
    weightBufferTensorDesc.Sizes = mWeightTensor->getShapeRef().data();
    weightBufferTensorDesc.Strides = weightStrides.data();
    weightBufferTensorDesc.GuaranteedBaseOffsetAlignment = 0;
    weightBufferTensorDesc.TotalTensorSizeInBytes =
        mWeightTensor->getTensorBufferSize();
  }

  DML_TENSOR_DESC weightTensorDesc;
  weightTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
  weightTensorDesc.Desc = &weightBufferTensorDesc;

  // START
  // create biasTensorDesc
  DML_BUFFER_TENSOR_DESC biasBufferTensorDesc;
  TensorSizes biasStrides;
  DML_TENSOR_DESC biasTensorDesc;
  if (biasData != nullptr) {
    biasStrides = dml_helper::CalculateStrides(mBiasTensor->getShapeRef());
    {
      // DML_TENSOR_FLAG_OWNED_BY_DML
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
  // END

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

  // create GEMM operator descriptor
  DML_GEMM_OPERATOR_DESC gemmOperatorDesc;
  {
    gemmOperatorDesc.ATensor = &inputTensorDesc;
    gemmOperatorDesc.BTensor = &weightTensorDesc;
    gemmOperatorDesc.CTensor = &biasTensorDesc; // nullptr;
    gemmOperatorDesc.OutputTensor = &outputTensorDesc;
    gemmOperatorDesc.TransA =
        DML_MATRIX_TRANSFORM_NONE; // DML_MATRIX_TRANSFORM_NONE
    gemmOperatorDesc.TransB =
        DML_MATRIX_TRANSFORM_NONE; // DML_MATRIX_TRANSFORM_TRANSPOSE
    gemmOperatorDesc.Alpha = 1.f;
    gemmOperatorDesc.Beta = 1.f;
    gemmOperatorDesc.FusedActivation = nullptr;
  }

  DML_OPERATOR_DESC operatorDesc{};
  operatorDesc.Type = DML_OPERATOR_GEMM;
  operatorDesc.Desc = &gemmOperatorDesc;

  DeviceManager &deviceManager = DeviceManager::getInstance();

  winrt::com_ptr<IDMLOperator> dmlOperator;
  winrt::check_hresult(deviceManager.mDMLDevice->CreateOperator(
      &operatorDesc, __uuidof(dmlOperator), dmlOperator.put_void()));

  winrt::check_hresult(deviceManager.mDMLDevice->CompileOperator(
      dmlOperator.get(), DML_EXECUTION_FLAG_NONE, __uuidof(mCompiledOperator),
      mCompiledOperator.put_void()));

  InitBindingTables();
}

std::shared_ptr<WeightTensor> FullyConnectionOperator::getWeightTensor() {
  return mWeightTensor;
}

void FullyConnectionOperator::InitBindingTables() {
  auto deafultDevice = DeviceManager::getInstance().GetDefault();

  winrt::com_ptr<IDMLOperatorInitializer> operatorInitializer;
  IDMLCompiledOperator *dmlCompiledOperators[] = {mCompiledOperator.get()};
  winrt::check_hresult(deafultDevice->mDMLDevice->CreateOperatorInitializer(
      1, dmlCompiledOperators, __uuidof(operatorInitializer),
      operatorInitializer.put_void()));

  {
    uint32_t descriptorCount =
        operatorInitializer->GetBindingProperties().RequiredDescriptorCount;

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    winrt::check_hresult(deafultDevice->mD3D12Device->CreateDescriptorHeap(
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
    winrt::check_hresult(deafultDevice->mD3D12Device->CreateDescriptorHeap(
        &descriptorHeapDesc, _uuidof(mExecDescriptorHeap),
        mExecDescriptorHeap.put_void()));
  }

  {
    auto bindingProp = mCompiledOperator->GetBindingProperties();

    if (bindingProp.PersistentResourceSize > 0) {
      D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
          bindingProp.PersistentResourceSize,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

      winrt::check_hresult(deafultDevice->mD3D12Device->CreateCommittedResource(
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
    deafultDevice->mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                                    descriptorHeaps);

    DML_BINDING_TABLE_DESC tableDesc = {
        operatorInitializer.get(),
        mInitDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
        mInitDescriptorHeap->GetGPUDescriptorHandleForHeapStart(),
        bindingProps.RequiredDescriptorCount};

    winrt::check_hresult(deafultDevice->mDMLDevice->CreateBindingTable(
        &tableDesc, __uuidof(mInitBindingTable), mInitBindingTable.put_void()));

    if (bindingProps.TemporaryResourceSize > 0) {
      assert("ConvolutionOperatorTmp initBindingTable cannot handle temporary "
             "resource size");
    }
  }

  // TODO: should be global constant
  const DML_BUFFER_BINDING emptyBufferBinding = {nullptr, 0, 0};
  const DML_BINDING_DESC emptyBindingDesc = {DML_BINDING_TYPE_NONE, nullptr};

  DML_BUFFER_BINDING initBufferBindings[4] = {
      emptyBufferBinding,
      {mWeightTensor->getBufferPtr(), 0,
       mWeightTensor->getBufferPtr()->GetDesc().Width},
      {mBiasTensor->getBufferPtr(), 0,
       mBiasTensor->getBufferPtr()->GetDesc().Width}};

  DML_BUFFER_ARRAY_BINDING initBufferArrayBindings = {3, initBufferBindings};
  DML_BINDING_DESC initBindings = {DML_BINDING_TYPE_BUFFER_ARRAY,
                                   &initBufferArrayBindings};

  mInitBindingTable->BindInputs(1, &initBindings);

  DML_BUFFER_BINDING persistentBufferBinding;
  DML_BINDING_DESC persistentBindingDesc;
  {
    if (mPersistentResource.get() != nullptr) {
      persistentBufferBinding = {mPersistentResource.get(), 0,
                                 mPersistentResource->GetDesc().Width};
      persistentBindingDesc = {DML_BINDING_TYPE_BUFFER,
                               &persistentBufferBinding};
    } else {
      persistentBindingDesc = emptyBindingDesc;
    }

    mInitBindingTable->BindOutputs(1, &persistentBindingDesc);
  }

  deafultDevice->mCommandRecorder->RecordDispatch(
      deafultDevice->mCommandList.get(), operatorInitializer.get(),
      mInitBindingTable.get());

  {
    auto bindingProps = mCompiledOperator->GetBindingProperties();

    ID3D12DescriptorHeap *descriptorHeaps[] = {mExecDescriptorHeap.get()};
    deafultDevice->mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                                    descriptorHeaps);

    DML_BINDING_TABLE_DESC tableDesc = {
        mCompiledOperator.get(),
        mExecDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
        mExecDescriptorHeap->GetGPUDescriptorHandleForHeapStart(),
        bindingProps.RequiredDescriptorCount};

    winrt::check_hresult(deafultDevice->mDMLDevice->CreateBindingTable(
        &tableDesc, __uuidof(mExecBindingTable), mExecBindingTable.put_void()));
  }

  deafultDevice->CloseExecuteResetWait();

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
    mExecBindingTable->BindPersistentResource(&persistentBindingDesc);
  }
}

} // namespace colombia_supremo
