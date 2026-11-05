/*
 * See-Through Bubble: runtime access to DXC (disassemble, assemble, validate).
 * Uses the game's own dxcompiler.dll / dxil.dll from the executable's folder,
 * so players don't need extra files. The DLL stays loaded for the process
 * lifetime (COM objects must not outlive it).
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <windows.h>

#include <dxcapi.h>

#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace dxc_runtime {

namespace internal {
template <typename T>
struct Ref {
  T* ptr = nullptr;
  ~Ref() {
    if (ptr != nullptr) ptr->Release();
  }
  T** operator&() { return &ptr; }
  T* operator->() const { return ptr; }
  explicit operator bool() const { return ptr != nullptr; }
};

inline std::once_flag load_once;
inline HMODULE library = nullptr;
inline DxcCreateInstanceProc create_instance = nullptr;
inline std::string load_error;

inline std::string ErrorText(IDxcOperationResult* result) {
  Ref<IDxcBlobEncoding> errors;
  if (FAILED(result->GetErrorBuffer(&errors)) || !errors || errors->GetBufferSize() == 0) return {};
  return {static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize()};
}
}  // namespace internal

// Loads dxcompiler.dll from the game's executable folder (once). DXC objects are
// created per call, so different threads can disassemble/assemble in parallel.
inline bool EnsureLoaded(std::string* error) {
  using namespace internal;
  std::call_once(load_once, [] {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const auto path = std::filesystem::path(exe).parent_path() / L"dxcompiler.dll";
    library = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (library == nullptr) {
      load_error = "could not load " + path.string();
      return;
    }
    create_instance = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(library, "DxcCreateInstance"));
    if (create_instance == nullptr) load_error = "DxcCreateInstance missing";
  });
  if (create_instance == nullptr && error != nullptr) *error = load_error;
  return create_instance != nullptr;
}

inline std::optional<std::string> Disassemble(std::span<const uint8_t> code, std::string* error) {
  using namespace internal;
  if (!EnsureLoaded(error)) return std::nullopt;
  Ref<IDxcLibrary> lib;
  Ref<IDxcCompiler> compiler;
  Ref<IDxcBlobEncoding> blob;
  Ref<IDxcBlobEncoding> text;
  if (FAILED(create_instance(CLSID_DxcLibrary, IID_PPV_ARGS(&lib)))
      || FAILED(create_instance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)))
      || FAILED(lib->CreateBlobWithEncodingOnHeapCopy(code.data(), static_cast<UINT32>(code.size()), 0, &blob))
      || FAILED(compiler->Disassemble(blob.ptr, &text)) || !text) {
    if (error) *error = "disassembly failed";
    return std::nullopt;
  }
  return std::string(static_cast<const char*>(text->GetBufferPointer()), text->GetBufferSize());
}

// Assembles LLVM IR text into a DXIL container and validates (signs) it.
inline std::optional<std::vector<uint8_t>> AssembleAndValidate(const std::string& ir, std::string* error) {
  using namespace internal;
  if (!EnsureLoaded(error)) return std::nullopt;
  Ref<IDxcLibrary> lib;
  Ref<IDxcBlobEncoding> source;
  Ref<IDxcAssembler> assembler;
  Ref<IDxcOperationResult> assembled;
  Ref<IDxcBlob> container;
  Ref<IDxcValidator> validator;
  Ref<IDxcOperationResult> validated;
  HRESULT status = S_OK;
  if (FAILED(create_instance(CLSID_DxcLibrary, IID_PPV_ARGS(&lib)))
      || FAILED(lib->CreateBlobWithEncodingOnHeapCopy(ir.data(), static_cast<UINT32>(ir.size()), CP_UTF8, &source))
      || FAILED(create_instance(CLSID_DxcAssembler, IID_PPV_ARGS(&assembler)))
      || FAILED(assembler->AssembleToContainer(source.ptr, &assembled))) {
    if (error) *error = "assembler unavailable";
    return std::nullopt;
  }
  if (FAILED(assembled->GetStatus(&status)) || FAILED(status) || FAILED(assembled->GetResult(&container)) || !container) {
    if (error) *error = "assembly failed: " + ErrorText(assembled.ptr);
    return std::nullopt;
  }
  if (FAILED(create_instance(CLSID_DxcValidator, IID_PPV_ARGS(&validator)))
      || FAILED(validator->Validate(container.ptr, DxcValidatorFlags_InPlaceEdit, &validated))) {
    if (error) *error = "validator unavailable";
    return std::nullopt;
  }
  if (FAILED(validated->GetStatus(&status)) || FAILED(status)) {
    if (error) *error = "validation failed: " + ErrorText(validated.ptr);
    return std::nullopt;
  }
  const auto* data = static_cast<const uint8_t*>(container->GetBufferPointer());
  return std::vector<uint8_t>(data, data + container->GetBufferSize());
}

}  // namespace dxc_runtime
