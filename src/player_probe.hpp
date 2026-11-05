/*
 * See-Through Bubble: probe for the player's position from the game's constants.
 *
 * Character shaders bind two global constant buffers:
 *  - GlobalMaterialGlobalParameter_Common (cb50, space98): _mainPosition (0),
 *    _mainPositionPrev (16), _headPosition (32), _hpPercentage (76).
 *  - SceneConstantBuffer (cb15, space35): _bufferSizeAndInvSize (64), _viewPos (128),
 *    _viewProjNoJitter (224), _viewProjRelativeNoJitter (352), _renderingOriginPos (2160),
 *    _viewTileRelativePos (2672).
 * This probe finds how they are bound (root CBV or descriptor table), copies
 * them on the GPU when the game binds them as root CBVs (once per present each),
 * reads the copies two presents later and projects _headPosition and
 * _mainPosition to the screen in two ways (absolute world, camera-relative). The
 * overlay shows dots so the user can see which one sits on the character.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <include/reshade.hpp>

namespace player_probe {

using namespace reshade::api;

constexpr int kGlobal = 0;  // GlobalMaterialGlobalParameter_Common
constexpr int kScene = 1;   // SceneConstantBuffer
constexpr uint32_t kRegister[2] = {50, 15};
constexpr uint32_t kSpace[2] = {98, 35};
constexpr uint64_t kBytes[2] = {80, 2768};
constexpr uint32_t kSceneCopies = 8;  // scene binds copied per present (shadow cameras come too)
constexpr uint32_t kGlobalCopies = 8;  // distinct global buffers copied per present (passes bind others at b50)
constexpr uint64_t kGlobalOffset = kSceneCopies * 2768;
constexpr uint64_t kStampOffset = kGlobalOffset + kGlobalCopies * 80;
constexpr uint32_t kStampCount = kSceneCopies + kGlobalCopies;  // one 8-byte stamp per copy
constexpr uint64_t kSlotBytes = kStampOffset + kStampCount * 8;  // scene copies, global copies, stamps
// Ring of readback slots, one per present. The GPU may run several presents behind
// (DLSS frame generation, Reflex queues), so each copy also copies a stamp (the
// present number, from a small upload buffer) and reads only use slots whose
// stamps show the GPU has done the copies: the newest such slot wins.
constexpr uint32_t kSlots = 8;
constexpr uint32_t kStampEntries = 64;  // upload ring of present numbers

inline void Log(reshade::log::level lvl, const std::string& text) {
  reshade::log::message(lvl, ("seethroughbubble probe: " + text).c_str());
}

// ------------------------------------------------------------ layout survey

inline std::shared_mutex layouts_mutex;
inline std::unordered_map<uint64_t, uint32_t> root_param[2];  // layout -> root CBV param index
inline std::atomic<uint32_t> layouts_root[2] = {};
inline std::atomic<uint32_t> layouts_table[2] = {};
// Every binding seen: (param type, descriptor type, space, register, count) -> occurrences.
inline std::mutex survey_mutex;
inline std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> survey;
inline uint32_t layouts_logged = 0;

inline void Note(pipeline_layout_param_type param_type, const descriptor_range& range) {
  std::scoped_lock lock(survey_mutex);
  survey[{static_cast<uint32_t>(param_type), static_cast<uint32_t>(range.type), range.dx_register_space,
          range.dx_register_index, range.count}]++;
}

// Ranges of a table-like param. Types 4 and 5 (with static samplers) use the
// larger descriptor_range_with_static_samplers, so they must be indexed as such.
// ReShade 6.8.0 reports some of the game's tables as type 4.
inline uint32_t RangeCount(const pipeline_layout_param& param) {
  switch (param.type) {
    case pipeline_layout_param_type::descriptor_table:
    case pipeline_layout_param_type::push_descriptors_with_ranges:
      return param.descriptor_table.count;
    case pipeline_layout_param_type::descriptor_table_with_static_samplers:
    case pipeline_layout_param_type::push_descriptors_with_static_samplers:
      return param.descriptor_table_with_static_samplers.ranges != nullptr
                 ? param.descriptor_table_with_static_samplers.count
                 : 0u;
    default: return 0u;
  }
}

inline const descriptor_range& RangeAt(const pipeline_layout_param& param, uint32_t r) {
  if (param.type == pipeline_layout_param_type::descriptor_table_with_static_samplers
      || param.type == pipeline_layout_param_type::push_descriptors_with_static_samplers) {
    return param.descriptor_table_with_static_samplers.ranges[r];
  }
  return param.descriptor_table.ranges[r];
}

// One layout written out: param type, then per range "descriptor type:space:register+count".
// Static samplers (type 5) are summarized as a count.
inline std::string DescribeLayout(uint32_t param_count, const pipeline_layout_param* params) {
  std::string text;
  char item[64];
  for (uint32_t i = 0; i < param_count; ++i) {
    const auto& param = params[i];
    std::snprintf(item, sizeof(item), "%s[%u]p%u", i ? " " : "", i, static_cast<uint32_t>(param.type));
    text += item;
    if (param.type == pipeline_layout_param_type::push_constants) {
      std::snprintf(item, sizeof(item), "(s%u:b%u n%u)", param.push_constants.dx_register_space,
                    param.push_constants.dx_register_index, param.push_constants.count);
      text += item;
    } else if (param.type == pipeline_layout_param_type::push_descriptors) {
      const auto& r = param.push_descriptors;
      std::snprintf(item, sizeof(item), "(t%u s%u:%u)", static_cast<uint32_t>(r.type), r.dx_register_space,
                    r.dx_register_index);
      text += item;
    } else if (param.type == pipeline_layout_param_type::push_descriptors_with_static_samplers) {
      std::snprintf(item, sizeof(item), "(%u samplers)", RangeCount(param));
      text += item;
    } else {
      const uint32_t count = RangeCount(param);
      text += "{";
      for (uint32_t r = 0; r < count; ++r) {
        const auto& range = RangeAt(param, r);
        std::snprintf(item, sizeof(item), "%st%u s%u:%u+%d@%u", r ? " " : "", static_cast<uint32_t>(range.type),
                      range.dx_register_space, range.dx_register_index,
                      range.count == UINT32_MAX ? -1 : static_cast<int>(range.count), range.binding);
        text += item;
      }
      text += "}";
    }
  }
  return text;
}

inline uint32_t layouts_with_target_logged = 0;  // under survey_mutex

inline bool Covers(const descriptor_range& range, int which) {
  return range.type == descriptor_type::constant_buffer && range.dx_register_space == kSpace[which]
         && range.dx_register_index <= kRegister[which]
         && kRegister[which] < range.dx_register_index + (std::max)(range.count, 1u);
}

// Where in a layout's descriptor tables the two buffers can sit. The global
// constants are cb50/space98; SceneConstantBuffer's register in space35 differs
// per shader (b15/b19/b20), so every CBV slot of a space35 range is a candidate
// and the bind picks the one whose CBV has the scene buffer's size.
constexpr uint32_t kMaxSceneSlots = 64;  // CBV slots per space35 range examined

struct TableTarget {
  uint32_t param = UINT32_MAX;
  std::vector<std::pair<uint32_t, uint32_t>> slots;  // (offset in table, register)
};
inline std::unordered_map<uint64_t, std::pair<TableTarget, TableTarget>> table_targets;  // under layouts_mutex

inline void NoteTableTargets(const pipeline_layout_param& param, uint32_t index, TableTarget (&targets)[2]) {
  for (uint32_t r = 0; r < RangeCount(param); ++r) {
    const auto& range = RangeAt(param, r);
    if (range.type != descriptor_type::constant_buffer) continue;
    if (Covers(range, kGlobal)) {
      targets[kGlobal].param = index;
      targets[kGlobal].slots.emplace_back(range.binding + (kRegister[kGlobal] - range.dx_register_index),
                                          kRegister[kGlobal]);
    } else if (range.dx_register_space == kSpace[kScene]) {
      targets[kScene].param = index;
      const uint32_t count = (std::min)(range.count, kMaxSceneSlots);
      for (uint32_t k = 0; k < count; ++k) {
        targets[kScene].slots.emplace_back(range.binding + k, range.dx_register_index + k);
      }
    }
  }
}

inline void OnInitPipelineLayout(device* device, uint32_t param_count, const pipeline_layout_param* params,
                                 pipeline_layout layout) {
  if (layout.handle == 0 || device->get_api() != device_api::d3d12) return;
  // Does any range or root descriptor sit in space 35 or 98 (any type)?
  bool has_target_space = false;
  for (uint32_t i = 0; i < param_count && !has_target_space; ++i) {
    const auto& param = params[i];
    if (param.type == pipeline_layout_param_type::push_descriptors) {
      const uint32_t s = param.push_descriptors.dx_register_space;
      has_target_space = s == kSpace[0] || s == kSpace[1];
    } else if (param.type == pipeline_layout_param_type::push_constants) {
      const uint32_t s = param.push_constants.dx_register_space;
      has_target_space = s == kSpace[0] || s == kSpace[1];
    } else {
      for (uint32_t r = 0; r < RangeCount(param) && !has_target_space; ++r) {
        const uint32_t s = RangeAt(param, r).dx_register_space;
        has_target_space = s == kSpace[0] || s == kSpace[1];
      }
    }
  }
  {
    std::scoped_lock lock(survey_mutex);
    // Log a sample of layouts in full: the first few, then every 50th, plus the
    // first layouts that use space 35 or 98.
    const uint32_t n = layouts_logged++;
    if (n < 8 || (n % 50 == 0 && n < 1000)) {
      Log(reshade::log::level::info, "layout #" + std::to_string(n) + ": " + DescribeLayout(param_count, params));
    } else if (has_target_space && layouts_with_target_logged < 6) {
      Log(reshade::log::level::info,
          "layout #" + std::to_string(n) + " (space 35/98): " + DescribeLayout(param_count, params));
    }
    if (has_target_space) layouts_with_target_logged++;
  }
  TableTarget targets[2];
  for (uint32_t i = 0; i < param_count; ++i) NoteTableTargets(params[i], i, targets);
  if (targets[kGlobal].param != UINT32_MAX || targets[kScene].param != UINT32_MAX) {
    std::unique_lock lock(layouts_mutex);
    table_targets[layout.handle] = {std::move(targets[kGlobal]), std::move(targets[kScene])};
  }
  for (uint32_t i = 0; i < param_count; ++i) {
    const auto& param = params[i];
    if (param.type == pipeline_layout_param_type::push_descriptors) {
      const auto& range = param.push_descriptors;
      Note(param.type, range);
      if (range.type != descriptor_type::constant_buffer) continue;
      for (int which = 0; which < 2; ++which) {
        if (!Covers(range, which)) continue;
        layouts_root[which]++;
        std::unique_lock lock(layouts_mutex);
        root_param[which][layout.handle] = i;
      }
    } else if (param.type == pipeline_layout_param_type::push_descriptors_with_static_samplers) {
      continue;  // static samplers only
    } else {
      for (uint32_t r = 0; r < RangeCount(param); ++r) {
        const auto& range = RangeAt(param, r);
        Note(param.type, range);
        if (range.type != descriptor_type::constant_buffer) continue;
        for (int which = 0; which < 2; ++which) {
          if (Covers(range, which)) layouts_table[which]++;
        }
      }
    }
  }
}

inline void OnDestroyPipelineLayout(device*, pipeline_layout layout) {
  std::unique_lock lock(layouts_mutex);
  root_param[0].erase(layout.handle);
  root_param[1].erase(layout.handle);
  table_targets.erase(layout.handle);
}

inline std::string SurveyText() {
  std::scoped_lock lock(survey_mutex);
  std::string text = "bindings in " + std::to_string(layouts_logged)
                     + " layouts (param type/descriptor type space:register+count x occurrences):";
  std::vector<std::pair<uint32_t, std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>>> sorted;
  for (const auto& [key, count] : survey) sorted.emplace_back(count, key);
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  for (size_t i = 0; i < sorted.size() && i < 60; ++i) {
    const auto& [count, key] = sorted[i];
    const auto [ptype, dtype, space, reg, n] = key;
    char item[64];
    std::snprintf(item, sizeof(item), " p%u/t%u s%u:%u+%d x%u", ptype, dtype, space, reg,
                  n == UINT32_MAX ? -1 : static_cast<int>(n), count);
    text += item;
  }
  return text;
}

// ------------------------------------------------------------ capture

inline std::mutex gpu_mutex;
inline device* gpu_device = nullptr;
inline resource readback = {0};
inline resource stamps = {0};  // upload heap: uint64 present number at (present % kStampEntries) * 8
inline bool gpu_failed = false;
inline std::atomic<uint64_t> present_count = 1;
inline std::atomic<uint64_t> captured_present[2] = {};
inline uint64_t slot_present[kSlots][2] = {};  // present whose copies went into this slot, per buffer
inline uint32_t slot_scene_count[kSlots] = {};
inline uint32_t slot_global_count[kSlots] = {};
inline std::pair<uint64_t, uint64_t> slot_global_source[kSlots][kGlobalCopies] = {};  // (buffer, offset)
inline std::atomic<uint32_t> binds_seen[2] = {};
inline std::atomic<uint32_t> no_buffer[2] = {};
inline std::atomic<uint32_t> copies[2] = {};
inline std::atomic<uint32_t> heap_logged[2] = {};
// Timing stats (log line every 600 presents): readings used, sum of their ages, presents without one.
inline std::atomic<uint32_t> timing_readings = 0, timing_age_sum = 0, timing_none = 0, timing_max_age = 0;

inline void WriteStamp(uint64_t present) {
  void* mapped = nullptr;
  const uint64_t offset = (present % kStampEntries) * 8;
  if (gpu_device->map_buffer_region(stamps, offset, 8, map_access::write_only, &mapped) && mapped != nullptr) {
    std::memcpy(mapped, &present, 8);
    gpu_device->unmap_buffer_region(stamps);
  }
}

inline bool EnsureReadback(device* dev) {
  if (gpu_failed) return false;
  if (readback.handle != 0) return gpu_device == dev;
  if (!dev->create_resource(resource_desc(kSlotBytes * kSlots, memory_heap::gpu_to_cpu, resource_usage::copy_dest),
                            nullptr, resource_usage::copy_dest, &readback)) {
    gpu_failed = true;
    Log(reshade::log::level::warning, "could not create readback buffer");
    return false;
  }
  if (!dev->create_resource(resource_desc(kStampEntries * 8, memory_heap::cpu_to_gpu, resource_usage::copy_source),
                            nullptr, resource_usage::cpu_access, &stamps)) {
    dev->destroy_resource(readback);
    readback = {0};
    gpu_failed = true;
    Log(reshade::log::level::warning, "could not create stamp buffer");
    return false;
  }
  gpu_device = dev;
  // Every entry starts as "never" (0); the current present gets its number now.
  void* mapped = nullptr;
  if (dev->map_buffer_region(stamps, 0, kStampEntries * 8, map_access::write_only, &mapped) && mapped != nullptr) {
    std::memset(mapped, 0, kStampEntries * 8);
    dev->unmap_buffer_region(stamps);
  }
  WriteStamp(present_count.load(std::memory_order_relaxed));
  return true;
}

inline void Capture(command_list* cmd_list, int which, const buffer_range& range, const char* how);

inline void OnPushDescriptors(command_list* cmd_list, shader_stage, pipeline_layout layout, uint32_t layout_param,
                              const descriptor_table_update& update) {
  if (update.type != descriptor_type::constant_buffer || update.count == 0 || update.binding != 0
      || update.descriptors == nullptr) {
    return;
  }
  int which = -1;
  {
    std::shared_lock lock(layouts_mutex);
    for (int w = 0; w < 2 && which < 0; ++w) {
      auto it = root_param[w].find(layout.handle);
      if (it != root_param[w].end() && it->second == layout_param) which = w;
    }
  }
  if (which < 0) return;
  binds_seen[which]++;
  Capture(cmd_list, which, *static_cast<const buffer_range*>(update.descriptors), "root CBV");
}

// ------------------------------------------------------------ table CBVs
//
// The game binds both buffers through descriptor tables ({CBV space35 0+N},
// {CBV space98 50+5}). D3D12 descriptors can't be read back, so CBV descriptors
// are tracked as they are written (CreateConstantBufferView ->
// update_descriptor_tables) and copied (CopyDescriptors* -> copy_descriptor_tables),
// keyed by (heap, offset). Only CBVs are stored.

constexpr uint32_t kMaxHeapSlots = 1u << 21;     // ignore offsets beyond this (D3D12 max is 1M shader-visible)
constexpr uint64_t kSceneCbvMin = 2768, kSceneCbvMax = 3072;  // SceneConstantBuffer, rounded to 256
constexpr uint64_t kGlobalCbvMin = 80;

inline std::shared_mutex cbv_mutex;
inline std::unordered_map<uint64_t, std::vector<buffer_range>> cbv_heaps;
inline std::atomic<uint32_t> cbv_written = 0, cbv_copied = 0, table_binds[2] = {}, table_misses[2] = {};
inline std::atomic<uint32_t> table_logged[2] = {};

inline bool HeapOffset(device* dev, descriptor_table table, uint32_t binding, uint64_t* heap, uint32_t* offset) {
  descriptor_heap h = {0};
  *offset = 0;
  dev->get_descriptor_heap_offset(table, binding, 0, &h, offset);
  *heap = h.handle;
  return h.handle != 0 && *offset < kMaxHeapSlots;
}

inline bool OnUpdateDescriptorTables(device* dev, uint32_t count, const descriptor_table_update* updates) {
  for (uint32_t i = 0; i < count; ++i) {
    const auto& update = updates[i];
    if (update.type != descriptor_type::constant_buffer || update.descriptors == nullptr || update.count == 0) continue;
    uint64_t heap;
    uint32_t offset;
    if (!HeapOffset(dev, update.table, update.binding + update.array_offset, &heap, &offset)) continue;
    std::unique_lock lock(cbv_mutex);
    auto& slots = cbv_heaps[heap];
    const uint32_t end = (std::min)(offset + update.count, kMaxHeapSlots);
    if (slots.size() < end) slots.resize(end);
    for (uint32_t k = 0; offset + k < end; ++k) slots[offset + k] = static_cast<const buffer_range*>(update.descriptors)[k];
    cbv_written += end - offset;
  }
  return false;
}

inline bool OnCopyDescriptorTables(device* dev, uint32_t count, const descriptor_table_copy* copies) {
  for (uint32_t i = 0; i < count; ++i) {
    const auto& copy = copies[i];
    if (copy.count == 0) continue;
    uint64_t src_heap, dst_heap;
    uint32_t src_offset, dst_offset;
    if (!HeapOffset(dev, copy.source_table, copy.source_binding + copy.source_array_offset, &src_heap, &src_offset)
        || !HeapOffset(dev, copy.dest_table, copy.dest_binding + copy.dest_array_offset, &dst_heap, &dst_offset)) {
      continue;
    }
    {
      // Fast path: neither heap ever held a CBV.
      std::shared_lock lock(cbv_mutex);
      if (!cbv_heaps.contains(src_heap) && !cbv_heaps.contains(dst_heap)) continue;
    }
    std::unique_lock lock(cbv_mutex);
    auto src_it = cbv_heaps.find(src_heap);
    const bool src_has = src_it != cbv_heaps.end() && src_offset < src_it->second.size();
    if (!src_has) {
      // Non-CBVs (or untracked) copied over: clear what we had there.
      auto dst_it = cbv_heaps.find(dst_heap);
      if (dst_it == cbv_heaps.end()) continue;
      auto& dst = dst_it->second;
      for (uint32_t k = 0; k < copy.count && dst_offset + k < dst.size(); ++k) dst[dst_offset + k] = {};
      continue;
    }
    // Copy into a temporary first: src and dst may be the same heap (and the vector may grow).
    const auto& src = src_it->second;
    std::vector<buffer_range> tmp(copy.count);
    for (uint32_t k = 0; k < copy.count && src_offset + k < src.size(); ++k) tmp[k] = src[src_offset + k];
    auto& dst = cbv_heaps[dst_heap];
    const uint32_t end = (std::min)(dst_offset + copy.count, kMaxHeapSlots);
    if (dst.size() < end) dst.resize(end);
    for (uint32_t k = 0; dst_offset + k < end; ++k) dst[dst_offset + k] = tmp[k];
    cbv_copied += end - dst_offset;
  }
  return false;
}

inline bool LookupCbv(device* dev, descriptor_table table, uint32_t binding, buffer_range* out) {
  uint64_t heap;
  uint32_t offset;
  if (!HeapOffset(dev, table, binding, &heap, &offset)) return false;
  std::shared_lock lock(cbv_mutex);
  auto it = cbv_heaps.find(heap);
  if (it == cbv_heaps.end() || offset >= it->second.size()) return false;
  *out = it->second[offset];
  return out->buffer.handle != 0;
}

inline void OnBindDescriptorTables(command_list* cmd_list, shader_stage, pipeline_layout layout, uint32_t first,
                                   uint32_t count, const descriptor_table* tables) {
  if (tables == nullptr || count == 0) return;
  TableTarget targets[2];
  {
    std::shared_lock lock(layouts_mutex);
    auto it = table_targets.find(layout.handle);
    if (it == table_targets.end()) return;
    const bool any = (it->second.first.param - first < count) || (it->second.second.param - first < count);
    if (!any) return;
    targets[kGlobal] = it->second.first;
    targets[kScene] = it->second.second;
  }
  const uint64_t present = present_count.load(std::memory_order_relaxed);
  auto* dev = cmd_list->get_device();
  for (int which = 0; which < 2; ++which) {
    const auto& target = targets[which];
    if (target.param - first >= count) continue;
    table_binds[which]++;
    if (captured_present[which].load(std::memory_order_relaxed) == present) continue;
    const descriptor_table table = tables[target.param - first];
    bool found = false;
    for (const auto& [slot, reg] : target.slots) {
      buffer_range range;
      if (!LookupCbv(dev, table, slot, &range)) continue;
      const uint64_t size = range.size == UINT64_MAX ? 0 : range.size;
      const bool fits = which == kGlobal ? size >= kGlobalCbvMin : (size >= kSceneCbvMin && size <= kSceneCbvMax);
      if (!fits) continue;
      found = true;
      if (table_logged[which]++ == 0) {
        Log(reshade::log::level::info, std::string(which == kGlobal ? "global material" : "scene")
                                           + " constants via table: param " + std::to_string(target.param) + ", b"
                                           + std::to_string(reg) + " (table offset " + std::to_string(slot)
                                           + "), CBV size " + std::to_string(size));
      }
      Capture(cmd_list, which, range, "table CBV");
      break;
    }
    if (!found) table_misses[which]++;
  }
}

inline void Capture(command_list* cmd_list, int which, const buffer_range& range, const char* how) {
  const uint64_t present = present_count.load(std::memory_order_relaxed);
  if (captured_present[which].load(std::memory_order_relaxed) == present) return;  // done for this present
  if (range.buffer.handle == 0) {
    no_buffer[which]++;
    return;
  }
  auto* dev = cmd_list->get_device();
  const auto desc = dev->get_resource_desc(range.buffer);
  if (desc.type != resource_type::buffer || range.offset + kBytes[which] > desc.buffer.size) return;
  std::scoped_lock lock(gpu_mutex);
  if (captured_present[which].load(std::memory_order_relaxed) == present) return;
  if (!EnsureReadback(dev)) return;
  const uint32_t slot = static_cast<uint32_t>(present % kSlots);
  if (slot_present[slot][which] != present) {
    slot_present[slot][which] = present;
    if (which == kScene) slot_scene_count[slot] = 0;
    if (which == kGlobal) slot_global_count[slot] = 0;
  }
  // Up to kSceneCopies scene binds (the main camera is picked on read by its aspect
  // ratio) and up to kGlobalCopies distinct global buffers (the player's is picked on
  // read by plausibility: some passes bind other data at b50).
  uint64_t dst = 0;
  uint32_t entry = 0;  // stamp index
  if (which == kScene) {
    entry = slot_scene_count[slot];
    dst = slot * kSlotBytes + slot_scene_count[slot] * kBytes[kScene];
    if (++slot_scene_count[slot] >= kSceneCopies) captured_present[kScene].store(present, std::memory_order_relaxed);
  } else {
    const std::pair<uint64_t, uint64_t> source(range.buffer.handle, range.offset);
    for (uint32_t i = 0; i < slot_global_count[slot]; ++i) {
      if (slot_global_source[slot][i] == source) return;  // already copied this present
    }
    slot_global_source[slot][slot_global_count[slot]] = source;
    entry = kSceneCopies + slot_global_count[slot];
    dst = slot * kSlotBytes + kGlobalOffset + slot_global_count[slot] * kBytes[kGlobal];
    if (++slot_global_count[slot] >= kGlobalCopies) captured_present[kGlobal].store(present, std::memory_order_relaxed);
  }
  if (heap_logged[which]++ == 0) {
    Log(reshade::log::level::info, std::string(which == kGlobal ? "global material" : "scene") + " constants: " + how + ", heap "
                                       + std::to_string(static_cast<int>(desc.heap)) + ", buffer size "
                                       + std::to_string(desc.buffer.size));
  }
  const bool upload = desc.heap == memory_heap::cpu_to_gpu;  // stays GENERIC_READ, no transitions allowed
  if (!upload) cmd_list->barrier(range.buffer, resource_usage::constant_buffer, resource_usage::copy_source);
  cmd_list->copy_buffer_region(range.buffer, range.offset, readback, dst, kBytes[which]);
  if (!upload) cmd_list->barrier(range.buffer, resource_usage::copy_source, resource_usage::constant_buffer);
  // The stamp after the data: it reads `present` once the GPU has done this copy.
  cmd_list->copy_buffer_region(stamps, (present % kStampEntries) * 8, readback, slot * kSlotBytes + kStampOffset + entry * 8, 8);
  copies[which]++;
}

// ------------------------------------------------------------ results

struct Projection {
  bool valid = false;
  float u = 0.f, v = 0.f, depth = 0.f, w = 0.f;
};

struct Result {
  bool valid = false;
  uint64_t present = 0;
  uint64_t position_present = 0;  // present of the player position (older when re-projected); `present` = camera
  float main_pos[3] = {}, head_pos[3] = {}, hp = 0.f;
  float view_pos[3] = {}, origin_pos[3] = {}, tile_pos[3] = {};
  float buffer_size[2] = {};
  Projection head_abs, head_rel, main_abs, main_rel;
  Projection body;  // torso point between main and head: what Auto mode follows
  float body_world[3] = {};  // that point, absolute world (metres, Y up)
  float velocity[3] = {};    // of the character, metres per present (smoothed; 0 after jumps)
  float inv_view_proj_relative[16] = {};
  uint32_t global_candidates = 0, global_plausible = 0;
  uint32_t rejected_aspect = 0;
};

constexpr float kBodyHeightFraction = 0.6f;  // torso point: main + 0.6 * (head - main)

// The player's global constants: finite, head above the feet, near the camera.
inline bool PlausiblePlayer(const float* main_pos, const float* head_pos, const float* view_pos) {
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(main_pos[i]) || !std::isfinite(head_pos[i])) return false;
  }
  const float up = head_pos[1] - main_pos[1];
  const float side = std::hypot(head_pos[0] - main_pos[0], head_pos[2] - main_pos[2]);
  const float dx = main_pos[0] - view_pos[0], dy = main_pos[1] - view_pos[1], dz = main_pos[2] - view_pos[2];
  return up > 0.3f && up < 3.f && side < 1.5f && dx * dx + dy * dy + dz * dz < 100.f * 100.f;
}

inline std::mutex result_mutex;
inline Result result;
inline std::atomic<uint32_t> rejected_aspect = 0;

// clip = M * (p, 1); M column_major as stored (column j at [4j .. 4j+3]), as HLSL mul(M, v).
inline Projection Project(const float* m, const float p[3]) {
  float clip[4] = {};
  const float v[4] = {p[0], p[1], p[2], 1.f};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) clip[i] += m[4 * j + i] * v[j];
  }
  Projection out;
  out.w = clip[3];
  if (!(std::abs(clip[3]) > 1e-6f) || !std::isfinite(clip[3])) return out;
  const float x = clip[0] / clip[3], y = clip[1] / clip[3];
  out.u = x * 0.5f + 0.5f;
  out.v = 0.5f - y * 0.5f;
  out.depth = clip[2] / clip[3];
  out.valid = clip[3] > 0.f && std::isfinite(out.u) && std::isfinite(out.v);
  return out;
}

// One slot as read back: the main-camera scene copy and the player's global copy,
// each only if its stamp shows the GPU has written it.
struct SlotReading {
  const uint8_t* scene = nullptr;
  const uint8_t* global = nullptr;
  uint32_t global_count = 0, global_plausible = 0;
};

inline SlotReading ReadSlot(const uint8_t* data, uint32_t slot, uint64_t want, uint32_t screen_width,
                            uint32_t screen_height) {
  SlotReading out;
  if (slot_present[slot][kScene] != want) return out;
  const uint8_t* base = data + slot * kSlotBytes;
  const auto stamp_ok = [&](uint32_t entry) {
    uint64_t stamp;
    std::memcpy(&stamp, base + kStampOffset + entry * 8, 8);
    return stamp == want;
  };
  // First scene copy with the screen's aspect ratio = main camera (shadow cameras differ).
  const float expected = screen_height != 0 ? static_cast<float>(screen_width) / static_cast<float>(screen_height) : 0.f;
  for (uint32_t i = 0; i < slot_scene_count[slot] && out.scene == nullptr; ++i) {
    if (!stamp_ok(i)) continue;
    const uint8_t* candidate = base + i * kBytes[kScene];
    float size[2];
    std::memcpy(size, candidate + 64, 8);
    if (size[1] > 0.f && expected > 0.f && std::abs(size[0] / size[1] / expected - 1.f) < 0.01f) {
      out.scene = candidate;
    } else {
      rejected_aspect++;
    }
  }
  if (out.scene == nullptr) return out;
  float view_pos[3];
  std::memcpy(view_pos, out.scene + 128, 12);
  // Of the global candidates, the plausible one nearest the camera.
  const uint32_t global_count = slot_present[slot][kGlobal] == want ? slot_global_count[slot] : 0;
  float best = 0.f;
  out.global_count = global_count;
  for (uint32_t i = 0; i < global_count; ++i) {
    if (!stamp_ok(kSceneCopies + i)) continue;
    const uint8_t* candidate = base + kGlobalOffset + i * kBytes[kGlobal];
    float main_pos[3], head_pos[3];
    std::memcpy(main_pos, candidate + 0, 12);
    std::memcpy(head_pos, candidate + 32, 12);
    if (!PlausiblePlayer(main_pos, head_pos, view_pos)) continue;
    out.global_plausible++;
    float d = 0.f;
    for (int k = 0; k < 3; ++k) d += (main_pos[k] - view_pos[k]) * (main_pos[k] - view_pos[k]);
    if (out.global == nullptr || d < best) {
      out.global = candidate;
      best = d;
    }
  }
  return out;
}

// reshade_present: use the newest slot the GPU has finished (stamps match).
inline void OnPresent(uint32_t screen_width, uint32_t screen_height) {
  std::scoped_lock lock(gpu_mutex);
  const uint64_t present = present_count.load(std::memory_order_relaxed) + 1;
  if (readback.handle != 0 && gpu_device != nullptr) WriteStamp(present);  // before any copy of this present
  present_count.store(present, std::memory_order_relaxed);
  if (readback.handle == 0 || gpu_device == nullptr || present < 2) return;
  void* mapped = nullptr;
  if (!gpu_device->map_buffer_region(readback, 0, kSlotBytes * kSlots, map_access::read_only, &mapped)
      || mapped == nullptr) {
    return;
  }
  const auto* mapped_data = static_cast<const uint8_t*>(mapped);

  // Newest finished slot with a player reading; else the newest finished camera.
  // Read in place (readback memory is CPU-cached) and copy out only the chosen slot.
  uint64_t want = 0;
  SlotReading reading;
  for (uint64_t back = 1; back < kSlots && back < present; ++back) {
    const uint64_t candidate_present = present - back;
    const auto slot_reading = ReadSlot(mapped_data, static_cast<uint32_t>(candidate_present % kSlots),
                                       candidate_present, screen_width, screen_height);
    if (slot_reading.scene == nullptr) continue;
    if (want == 0) {
      want = candidate_present;
      reading = slot_reading;
    }
    if (slot_reading.global != nullptr) {
      want = candidate_present;
      reading = slot_reading;
      break;
    }
  }
  if (want != 0) {
    static uint8_t data[kSlotBytes];  // only used under gpu_mutex
    const uint8_t* slot_base = mapped_data + (want % kSlots) * kSlotBytes;
    std::memcpy(data, slot_base, kSlotBytes);
    reading.scene = data + (reading.scene - slot_base);
    if (reading.global != nullptr) reading.global = data + (reading.global - slot_base);
  }
  gpu_device->unmap_buffer_region(readback);
  if (want == 0 || reading.global == nullptr) {
    timing_none++;
  } else {
    const uint32_t age = static_cast<uint32_t>(present - want);
    timing_readings++;
    timing_age_sum += age;
    if (age > timing_max_age.load(std::memory_order_relaxed)) timing_max_age = age;
  }
  if (want == 0) return;
  const uint8_t* scene = reading.scene;
  const uint8_t* global = reading.global;
  Result r;
  r.present = want;
  std::memcpy(r.buffer_size, scene + 64, 8);
  std::memcpy(r.view_pos, scene + 128, 12);
  r.global_candidates = reading.global_count;
  r.global_plausible = reading.global_plausible;
  // No plausible reading this present: re-project the last known position with
  // this present's camera (the character rarely moves far in a few frames; the
  // add-on stops after "Forget After" seconds).
  static float last_main[3], last_head[3], last_hp;  // under gpu_mutex
  static uint64_t last_position_present = 0;
  static float velocity[3] = {};  // metres per present, smoothed; under gpu_mutex
  if (global != nullptr && want <= last_position_present) global = nullptr;  // not newer: keep the last one
  if (global != nullptr) {
    float main_now[3];
    std::memcpy(main_now, global + 0, 12);
    const uint64_t steps = last_position_present != 0 ? want - last_position_present : 0;
    float v[3] = {}, speed2 = 0.f;
    for (int i = 0; i < 3 && steps != 0; ++i) {
      v[i] = (main_now[i] - last_main[i]) / static_cast<float>(steps);
      speed2 += v[i] * v[i];
    }
    // Plausible motion only (< 1 m per present over at most 8 presents); else teleport/respawn: no prediction.
    const bool sane = steps != 0 && steps <= 8 && speed2 < 1.f;
    for (int i = 0; i < 3; ++i) velocity[i] = sane ? 0.5f * velocity[i] + 0.5f * v[i] : 0.f;
    std::memcpy(last_main, global + 0, 12);
    std::memcpy(last_head, global + 32, 12);
    std::memcpy(&last_hp, global + 76, 4);
    last_position_present = want;
  } else if (last_position_present == 0) {
    std::scoped_lock rlock(result_mutex);
    result.global_candidates = r.global_candidates;
    result.global_plausible = 0;
    return;
  }
  std::memcpy(r.main_pos, last_main, 12);
  std::memcpy(r.head_pos, last_head, 12);
  r.hp = last_hp;
  r.position_present = last_position_present;
  std::memcpy(r.velocity, velocity, sizeof(velocity));
  for (int i = 0; i < 3; ++i) r.body_world[i] = r.main_pos[i];
  r.body_world[1] += kBodyHeightFraction * (r.head_pos[1] - r.main_pos[1]);
  std::memcpy(r.origin_pos, scene + 2160, 12);
  std::memcpy(r.tile_pos, scene + 2672, 12);
  float view_proj[16], view_proj_relative[16];
  std::memcpy(view_proj, scene + 224, 64);           // _viewProjNoJitter
  std::memcpy(view_proj_relative, scene + 352, 64);  // _viewProjRelativeNoJitter
  r.head_abs = Project(view_proj, r.head_pos);
  r.main_abs = Project(view_proj, r.main_pos);
  float rel[3];
  for (int i = 0; i < 3; ++i) rel[i] = r.head_pos[i] - r.view_pos[i];
  r.head_rel = Project(view_proj_relative, rel);
  for (int i = 0; i < 3; ++i) rel[i] = r.main_pos[i] - r.view_pos[i];
  r.main_rel = Project(view_proj_relative, rel);
  // Straight above the feet (Y up): the head's sideways sway in animations doesn't move it.
  for (int i = 0; i < 3; ++i) rel[i] = r.main_pos[i] - r.view_pos[i];
  rel[1] += kBodyHeightFraction * (r.head_pos[1] - r.main_pos[1]);
  r.body = Project(view_proj_relative, rel);
  std::memcpy(r.inv_view_proj_relative, scene + 480, 64);  // _invViewProjRelative
  r.valid = true;
  r.rejected_aspect = rejected_aspect.load();
  std::scoped_lock rlock(result_mutex);
  result = r;
}

inline Result GetResult() {
  std::scoped_lock lock(result_mutex);
  return result;
}

// Log line: how old the readings were (in presents) over the last period; resets.
inline std::string TimingText() {
  const uint32_t n = timing_readings.exchange(0), sum = timing_age_sum.exchange(0), none = timing_none.exchange(0),
                 max_age = timing_max_age.exchange(0);
  char text[160];
  std::snprintf(text, sizeof(text), "player position readings %u (age avg %.2f, max %u presents), presents without %u", n,
                n != 0 ? static_cast<float>(sum) / static_cast<float>(n) : 0.f, max_age, none);
  return text;
}

inline uint64_t PresentsSince(const Result& r) {
  return r.present == 0 ? UINT64_MAX : present_count.load(std::memory_order_relaxed) - r.present;
}

inline uint64_t PresentsSincePosition(const Result& r) {
  return r.position_present == 0 ? UINT64_MAX : present_count.load(std::memory_order_relaxed) - r.position_present;
}

// The body point predicted for the frame recorded next: the reading is a few presents
// old, so move it on by the character's velocity (capped at kMaxPredictPresents).
constexpr uint64_t kPredictExtraPresents = 1;  // the next frame is recorded after this present
constexpr uint64_t kMaxPredictPresents = 10;
inline void PredictBody(const Result& r, float out[3]) {
  const uint64_t age = PresentsSincePosition(r);
  const float steps = static_cast<float>((std::min)(age == UINT64_MAX ? 0 : age + kPredictExtraPresents, kMaxPredictPresents));
  for (int i = 0; i < 3; ++i) out[i] = r.body_world[i] + r.velocity[i] * steps;
}

inline std::string ProjText(const char* name, const Projection& p) {
  char text[96];
  if (!p.valid) {
    std::snprintf(text, sizeof(text), "%s: off (w %.3g)", name, p.w);
  } else {
    std::snprintf(text, sizeof(text), "%s: %.1f%% / %.1f%%, depth %.5f", name, p.u * 100.f, p.v * 100.f, p.depth);
  }
  return text;
}

inline std::string StatusText() {
  char head[320];
  std::snprintf(head, sizeof(head),
                "Player probe: layouts global root %u / table %u, scene root %u / table %u; binds %u / %u, copies %u / %u, "
                "no buffer %u / %u, other cameras %u",
                layouts_root[kGlobal].load(), layouts_table[kGlobal].load(), layouts_root[kScene].load(),
                layouts_table[kScene].load(), binds_seen[kGlobal].load(), binds_seen[kScene].load(), copies[kGlobal].load(),
                copies[kScene].load(), no_buffer[kGlobal].load(), no_buffer[kScene].load(), rejected_aspect.load());
  std::string text = head;
  std::snprintf(head, sizeof(head),
                "\n  tables: binds %u / %u, misses %u / %u; CBV descriptors written %u, copied %u, heaps %zu",
                table_binds[kGlobal].load(), table_binds[kScene].load(), table_misses[kGlobal].load(),
                table_misses[kScene].load(), cbv_written.load(), cbv_copied.load(),
                [] { std::shared_lock lock(cbv_mutex); return cbv_heaps.size(); }());
  text += head;
  const Result r = GetResult();
  if (!r.valid) return text + "\n  no reading yet";
  char line[400];
  std::snprintf(line, sizeof(line),
                "\n  head %.2f %.2f %.2f | main %.2f %.2f %.2f | hp %.2f\n  view %.2f %.2f %.2f | origin %.2f %.2f %.2f | "
                "tile %.2f %.2f %.2f | render %.0fx%.0f",
                r.head_pos[0], r.head_pos[1], r.head_pos[2], r.main_pos[0], r.main_pos[1], r.main_pos[2], r.hp,
                r.view_pos[0], r.view_pos[1], r.view_pos[2], r.origin_pos[0], r.origin_pos[1], r.origin_pos[2],
                r.tile_pos[0], r.tile_pos[1], r.tile_pos[2], r.buffer_size[0], r.buffer_size[1]);
  text += line;
  std::snprintf(line, sizeof(line), "\n  global candidates %u, plausible %u; camera %llu, position %llu presents old",
                r.global_candidates, r.global_plausible, static_cast<unsigned long long>(PresentsSince(r)),
                static_cast<unsigned long long>(PresentsSincePosition(r)));
  text += line;
  text += "\n  " + ProjText("body (followed)", r.body);
  text += "\n  " + ProjText("head abs (magenta)", r.head_abs) + "\n  " + ProjText("head rel (cyan)", r.head_rel);
  text += "\n  " + ProjText("main abs (pink)", r.main_abs) + "\n  " + ProjText("main rel (blue)", r.main_rel);
  return text;
}

inline void Use(DWORD fdw_reason) {
  switch (fdw_reason) {
    case DLL_PROCESS_ATTACH:
      reshade::register_event<reshade::addon_event::init_pipeline_layout>(OnInitPipelineLayout);
      reshade::register_event<reshade::addon_event::destroy_pipeline_layout>(OnDestroyPipelineLayout);
      reshade::register_event<reshade::addon_event::push_descriptors>(OnPushDescriptors);
      reshade::register_event<reshade::addon_event::bind_descriptor_tables>(OnBindDescriptorTables);
      reshade::register_event<reshade::addon_event::update_descriptor_tables>(OnUpdateDescriptorTables);
      reshade::register_event<reshade::addon_event::copy_descriptor_tables>(OnCopyDescriptorTables);
      break;
    case DLL_PROCESS_DETACH:
      reshade::unregister_event<reshade::addon_event::init_pipeline_layout>(OnInitPipelineLayout);
      reshade::unregister_event<reshade::addon_event::destroy_pipeline_layout>(OnDestroyPipelineLayout);
      reshade::unregister_event<reshade::addon_event::push_descriptors>(OnPushDescriptors);
      reshade::unregister_event<reshade::addon_event::bind_descriptor_tables>(OnBindDescriptorTables);
      reshade::unregister_event<reshade::addon_event::update_descriptor_tables>(OnUpdateDescriptorTables);
      reshade::unregister_event<reshade::addon_event::copy_descriptor_tables>(OnCopyDescriptorTables);
      break;
    default: break;
  }
}

}  // namespace player_probe
