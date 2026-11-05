/*
 * See-Through Bubble: minimal DXIL container reader.
 * Reads a shader's entry-point name from the PSV0 part so pipelines can be
 * classified by name (e.g. "vsStandardDepthPre") instead of by hash, which
 * survives game patches that recompile shaders.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

namespace dxil {

namespace internal {
inline bool Read32(std::span<const uint8_t> data, size_t offset, uint32_t* value) {
  if (offset + 4 > data.size()) return false;
  std::memcpy(value, data.data() + offset, 4);
  return true;
}
}  // namespace internal

// Returns the entry-point name, or an empty view if the container has none.
// The view points into `code`.
inline std::string_view EntryPointName(std::span<const uint8_t> code) {
  using internal::Read32;
  if (code.size() < 32 || std::memcmp(code.data(), "DXBC", 4) != 0) return {};

  uint32_t part_count = 0;
  if (!Read32(code, 28, &part_count)) return {};
  for (uint32_t i = 0; i < part_count; ++i) {
    uint32_t part_offset = 0;
    if (!Read32(code, 32 + (4 * i), &part_offset)) return {};
    if (part_offset + 8 > code.size()) return {};
    if (std::memcmp(code.data() + part_offset, "PSV0", 4) != 0) continue;

    uint32_t part_size = 0;
    Read32(code, part_offset + 4, &part_size);
    const size_t part_end = part_offset + 8 + static_cast<size_t>(part_size);
    if (part_end > code.size()) return {};
    const auto part = code.subspan(0, part_end);

    size_t p = part_offset + 8;
    uint32_t runtime_info_size = 0;
    if (!Read32(part, p, &runtime_info_size)) return {};
    p += 4;
    const size_t runtime_info = p;
    p += runtime_info_size;

    uint32_t resource_count = 0;
    if (!Read32(part, p, &resource_count)) return {};
    p += 4;
    if (resource_count != 0) {
      uint32_t bind_info_size = 0;
      if (!Read32(part, p, &bind_info_size)) return {};
      p += 4 + (static_cast<size_t>(bind_info_size) * resource_count);
    }

    // EntryFunctionName is the last field of PSVRuntimeInfo3 (52 bytes).
    if (runtime_info_size < 52) return {};
    uint32_t string_table_size = 0;
    if (!Read32(part, p, &string_table_size)) return {};
    p += 4;
    if (p + string_table_size > part.size()) return {};

    uint32_t name_offset = 0;
    if (!Read32(part, runtime_info + 48, &name_offset)) return {};
    if (name_offset >= string_table_size) return {};

    const auto* table = reinterpret_cast<const char*>(part.data() + p);
    const auto* name = table + name_offset;
    const size_t max_length = string_table_size - name_offset;
    return {name, strnlen(name, max_length)};
  }
  return {};
}

}  // namespace dxil
