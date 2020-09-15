// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <winrt/base.h>

#include "d3dx12.h"

namespace d3d12_helper {

winrt::com_ptr<ID3D12Device> CreateDevice();

winrt::com_ptr<ID3D12CommandQueue>
CreateCommandQueue(winrt::com_ptr<ID3D12Device> device);

winrt::com_ptr<ID3D12CommandAllocator>
CreateCommandAllocator(winrt::com_ptr<ID3D12Device> device);

winrt::com_ptr<ID3D12GraphicsCommandList>
CreateCommandList(winrt::com_ptr<ID3D12Device> device,
                  winrt::com_ptr<ID3D12CommandAllocator> commandAllocator);

void CloseExecuteResetWait(
    winrt::com_ptr<ID3D12Device> d3D12Device,
    winrt::com_ptr<ID3D12CommandQueue> commandQueue,
    winrt::com_ptr<ID3D12CommandAllocator> commandAllocator,
    winrt::com_ptr<ID3D12GraphicsCommandList> commandList);

} // namespace d3d12_helper
