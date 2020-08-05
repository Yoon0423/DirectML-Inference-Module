// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <dxgi1_4.h>

#include "Direct3D12HelperFunction.hpp"

#if defined(_DEBUG)

#include <iostream>

#endif

namespace d3d12_helper {

void InitializeDirect3D12(
    winrt::com_ptr<ID3D12Device> &d3D12Device,
    winrt::com_ptr<ID3D12CommandQueue> &commandQueue,
    winrt::com_ptr<ID3D12CommandAllocator> &commandAllocator,
    winrt::com_ptr<ID3D12GraphicsCommandList> &commandList) {
#if defined(_DEBUG)
  winrt::com_ptr<ID3D12Debug> d3D12Debug;
  if (FAILED(D3D12GetDebugInterface(__uuidof(d3D12Debug),
                                    d3D12Debug.put_void()))) {
    // The D3D12 debug layer is missing - you must install the Graphics Tools
    // optional feature
    winrt::throw_hresult(DXGI_ERROR_SDK_COMPONENT_MISSING);
  }
  d3D12Debug->EnableDebugLayer();
#endif

  winrt::com_ptr<IDXGIFactory4> dxgiFactory;
  winrt::check_hresult(
      CreateDXGIFactory1(__uuidof(dxgiFactory), dxgiFactory.put_void()));

  winrt::com_ptr<IDXGIAdapter> dxgiAdapter;
  UINT adapterIndex{};
  HRESULT hr{};
  do {
    dxgiAdapter = nullptr;
    ++adapterIndex; // TODO: add to choose NVIDIA GPU on my laptop. refactor it
                    // to implement more properly
    winrt::check_hresult(
        dxgiFactory->EnumAdapters(adapterIndex, dxgiAdapter.put()));
    ++adapterIndex;

    DXGI_ADAPTER_DESC pDesc;
    dxgiAdapter->GetDesc(&pDesc);

#if defined(_DEBUG)
    std::wcout << L"chosen GPU: " << pDesc.Description << L"\n";
#endif

    hr = ::D3D12CreateDevice(dxgiAdapter.get(), D3D_FEATURE_LEVEL_11_0,
                             __uuidof(d3D12Device), d3D12Device.put_void());
    if (hr == DXGI_ERROR_UNSUPPORTED)
      continue;
    winrt::check_hresult(hr);
  } while (hr != S_OK);

  D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
  commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

  winrt::check_hresult(d3D12Device->CreateCommandQueue(
      &commandQueueDesc, __uuidof(commandQueue), commandQueue.put_void()));

  winrt::check_hresult(d3D12Device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(commandAllocator),
      commandAllocator.put_void()));

  winrt::check_hresult(d3D12Device->CreateCommandList(
      0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.get(), nullptr,
      __uuidof(commandList), commandList.put_void()));
}

void CloseExecuteResetWait(
    winrt::com_ptr<ID3D12Device> d3D12Device,
    winrt::com_ptr<ID3D12CommandQueue> commandQueue,
    winrt::com_ptr<ID3D12CommandAllocator> commandAllocator,
    winrt::com_ptr<ID3D12GraphicsCommandList> commandList) {
  winrt::check_hresult(commandList->Close());

  ID3D12CommandList *commandLists[] = {commandList.get()};
  commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

  winrt::check_hresult(commandList->Reset(commandAllocator.get(), nullptr));

  winrt::com_ptr<ID3D12Fence> d3D12Fence;
  winrt::check_hresult(d3D12Device->CreateFence(
      0, D3D12_FENCE_FLAG_NONE, _uuidof(d3D12Fence), d3D12Fence.put_void()));

  winrt::handle fenceEventHandle{0};
  fenceEventHandle.attach(::CreateEvent(nullptr, true, false, nullptr));
  winrt::check_bool(bool{fenceEventHandle});

  winrt::check_hresult(
      d3D12Fence->SetEventOnCompletion(1, fenceEventHandle.get()));

  winrt::check_hresult(commandQueue->Signal(d3D12Fence.get(), 1));

  ::WaitForSingleObjectEx(fenceEventHandle.get(), INFINITE, FALSE);
}

} // namespace d3d12_helper
