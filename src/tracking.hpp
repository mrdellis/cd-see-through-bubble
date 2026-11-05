/*
 * See-Through Bubble: character tracking (Phase 4).
 *
 * 1. Learn which stencil values mark characters: when the game draws with a
 *    character pipeline (Character.hlsl entry points), record its stencil
 *    reference (masked by the pipeline's stencil write mask).
 * 2. Copy the scene depth-stencil when the game clears it (it is then in a
 *    known state, DEPTH_WRITE), so the copy holds the previous frame.
 * 3. At present, run bubble_track.cs on ReShade's command list over the copy:
 *    character pixels around the hint position are binned by depth.
 * 4. Two frames later, read the result back on the CPU. Pixels are grouped by
 *    stencil value and depth band; pick the group whose centroid is nearest to
 *    the home point (and the last position), merge groups close to it (one
 *    character uses several stencil values) and keep the centroid (screen
 *    position) and front-most depth. When nothing is found, the last result is
 *    held. Grouping by value matters: the game draws some props (clay jars)
 *    with character pipelines, at the same depth as the character.
 * The bubble shaders turn (center, depth) into a distance in metres themselves.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <span>
#include <string>

#include <include/reshade.hpp>

namespace tracking {

using namespace reshade::api;

// ------------------------------------------------------------ learned facts

inline std::atomic<uint32_t> character_set[8] = {};
inline std::atomic<uint32_t> character_write_mask = 0;
inline std::atomic<uint64_t> scene_depth = 0;  // resource handle of the scene depth-stencil

// Returns true when the value is new.
inline bool RecordCharacterStencil(uint8_t reference, uint8_t write_mask) {
  if (write_mask == 0) return false;
  const uint8_t value = reference & write_mask;
  const uint32_t bit = 1u << (value & 31);
  if (character_set[value >> 5].load(std::memory_order_relaxed) & bit) return false;
  const uint32_t before = character_set[value >> 5].fetch_or(bit, std::memory_order_relaxed);
  character_write_mask.fetch_or(write_mask, std::memory_order_relaxed);
  return (before & bit) == 0;
}

inline bool IsCharacterStencil(uint32_t stencil) {
  const uint32_t key = stencil & character_write_mask.load(std::memory_order_relaxed) & 0xFF;
  return (character_set[key >> 5].load(std::memory_order_relaxed) >> (key & 31)) & 1;
}

inline void NoteSceneDepthView(device* device, resource_view dsv) {
  if (dsv.handle == 0) return;
  const auto res = device->get_resource_from_view(dsv);
  if (res.handle != 0) scene_depth.store(res.handle, std::memory_order_relaxed);
}

// ------------------------------------------------------------ results

struct Result {
  bool valid = false;  // ever found
  bool found_now = false;
  float u = 0.5f, v = 0.5f;  // screen position 0..1
  float depth = 0.f;        // raw reversed-Z of the character's front-most cluster pixel
  uint32_t pixels = 0;
  uint32_t total_character_pixels = 0;
  uint32_t hint_stencil = 0;
  float hint_depth = 0.f;
  uint32_t frames_since_found = 0;
  float seconds_since_found = 0.f;
  std::chrono::steady_clock::time_point last_found = {};
  uint32_t copies = 0;
  uint32_t dispatches = 0;
  uint32_t top_stencil[6][2] = {};  // most common stencil values in the search area: {value, pixel count}
  float search_u = 0.5f, search_v = 0.5f;  // search center used for the last scan
  uint32_t chosen_values[8] = {};  // 256-bit set of the stencil values in the chosen cluster
};

inline std::mutex result_mutex;
inline Result result;

inline Result GetResult() {
  std::scoped_lock lock(result_mutex);
  return result;
}

// Tuning (set by the add-on).
inline float search_radius = 0.22f;  // fraction of screen height
// The search center may move at most this far (fraction of screen height) from
// the home point (Manual Center): the follow camera keeps the character near it,
// so tracking can't wander off and latch onto distant objects.
inline float home_leash = 0.22f;
inline float smoothing = 0.35f;      // 0..1 per readback; higher follows faster
inline uint32_t min_pixels = 40;
// Cluster choice: cost = distance to home + last_weight * distance to the last
// position (screen-height units). Groups within merge_radius of the chosen
// group's centroid (and within one depth band) join it.
inline float last_weight = 0.5f;
inline float merge_radius = 0.10f;

// ------------------------------------------------------------ GPU side

// Output layout (uint offsets); keep in sync with bubble_track.cs.
constexpr uint32_t kOutBoxMinX = 4;
constexpr uint32_t kOutBoxMinY = 5;
constexpr uint32_t kOutBuckets = 8;
constexpr uint32_t kSlotCount = 16;
constexpr uint32_t kBucketCount = 64;
constexpr uint32_t kBucketFields = 4;  // count, sum_x, sum_y (relative to box_min), max_depth
constexpr uint32_t kOutHistogram = kOutBuckets + kSlotCount * kBucketCount * kBucketFields;
constexpr uint32_t kOutputUints = kOutHistogram + 256;
constexpr uint32_t kOutputBytes = kOutputUints * 4;
constexpr uint32_t kReadbackSlots = 3;

struct TrackConstants {
  uint32_t size[2];
  int32_t box_min[2];
  float hint_px[2];
  float radius_px;
  uint32_t write_mask;
  uint32_t frame_id;
  uint32_t mode;
  uint32_t pad[2];
  uint32_t character_set[8];
};
static_assert(sizeof(TrackConstants) == 20 * 4);  // matches bubble_track.cs (uint4 character_set[2] at offset 48)

struct Gpu {
  device* dev = nullptr;
  bool failed = false;
  resource copy = {0};
  uint32_t width = 0, height = 0;
  resource output = {0};
  resource readback[kReadbackSlots] = {};
  uint32_t readback_frame[kReadbackSlots] = {};
  uint32_t readback_set[kReadbackSlots][8] = {};  // character set used by that scan (slot -> value mapping)
  ID3D12DescriptorHeap* heap = nullptr;  // non-shader-visible: SRV depth, SRV stencil, UAV output
  UINT descriptor_size = 0;
  pipeline_layout layout = {0};
  pipeline pipe = {0};
  bool copy_valid = false;
  uint32_t frame = 0;
};

inline std::mutex gpu_mutex;
inline Gpu gpu;
inline std::span<const uint8_t> compute_shader;  // set by the add-on

inline void Log(reshade::log::level lvl, const std::string& text) {
  reshade::log::message(lvl, ("seethroughbubble tracking: " + text).c_str());
}

inline D3D12_CPU_DESCRIPTOR_HANDLE Descriptor(uint32_t index) {
  D3D12_CPU_DESCRIPTOR_HANDLE handle = gpu.heap->GetCPUDescriptorHandleForHeapStart();
  handle.ptr += static_cast<SIZE_T>(index) * gpu.descriptor_size;
  return handle;
}

// Creates the copy texture, output/readback buffers, native views and the
// compute pipeline. Call with gpu_mutex held.
inline bool EnsureGpu(device* dev, const resource_desc& depth_desc) {
  if (gpu.failed) return false;
  const uint32_t width = depth_desc.texture.width;
  const uint32_t height = depth_desc.texture.height;
  if (gpu.dev == dev && gpu.copy.handle != 0 && gpu.width == width && gpu.height == height) return true;
  if (depth_desc.texture.depth_or_layers != 1 || depth_desc.texture.levels != 1 || depth_desc.texture.samples != 1
      || (depth_desc.texture.format != format::d32_float_s8_uint && depth_desc.texture.format != format::r32_g8_typeless)) {
    gpu.failed = true;
    Log(reshade::log::level::warning, "unsupported scene depth format/shape");
    return false;
  }
  if (gpu.copy.handle != 0) dev->destroy_resource(gpu.copy);
  gpu.copy = {0};
  gpu.copy_valid = false;
  gpu.dev = dev;

  auto* native = reinterpret_cast<ID3D12Device*>(dev->get_native());
  if (!dev->create_resource(resource_desc(width, height, 1, 1, format::r32_g8_typeless, 1, memory_heap::gpu_only,
                                          resource_usage::copy_dest | resource_usage::shader_resource),
                            nullptr, resource_usage::copy_dest, &gpu.copy)) {
    gpu.failed = true;
    Log(reshade::log::level::warning, "could not create depth copy");
    return false;
  }
  gpu.width = width;
  gpu.height = height;

  if (gpu.output.handle == 0) {
    if (!dev->create_resource(resource_desc(kOutputBytes, memory_heap::gpu_only,
                                            resource_usage::unordered_access | resource_usage::copy_source),
                              nullptr, resource_usage::unordered_access, &gpu.output)) {
      gpu.failed = true;
      return false;
    }
    for (auto& rb : gpu.readback) {
      if (!dev->create_resource(resource_desc(kOutputBytes, memory_heap::gpu_to_cpu, resource_usage::copy_dest), nullptr,
                                resource_usage::copy_dest, &rb)) {
        gpu.failed = true;
        return false;
      }
    }
  }

  if (gpu.heap == nullptr) {
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(native->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&gpu.heap)))) {
      gpu.failed = true;
      return false;
    }
    gpu.descriptor_size = native->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  }

  // ReShade can't create plane-1 (stencil) views, so the views are native.
  auto* copy_native = reinterpret_cast<ID3D12Resource*>(gpu.copy.handle);
  D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
  srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv.Texture2D.MipLevels = 1;
  srv.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
  srv.Texture2D.PlaneSlice = 0;
  native->CreateShaderResourceView(copy_native, &srv, Descriptor(0));
  srv.Format = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
  srv.Texture2D.PlaneSlice = 1;
  native->CreateShaderResourceView(copy_native, &srv, Descriptor(1));
  D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
  uav.Format = DXGI_FORMAT_R32_TYPELESS;
  uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uav.Buffer.NumElements = kOutputBytes / 4;
  uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
  native->CreateUnorderedAccessView(reinterpret_cast<ID3D12Resource*>(gpu.output.handle), nullptr, &uav, Descriptor(2));

  if (gpu.pipe.handle == 0) {
    constant_range constants = {};
    constants.count = sizeof(TrackConstants) / 4;
    constants.visibility = shader_stage::compute;
    descriptor_range srvs = {};
    srvs.count = 2;
    srvs.array_size = 1;
    srvs.type = descriptor_type::texture_shader_resource_view;
    srvs.visibility = shader_stage::compute;
    descriptor_range uavs = {};
    uavs.count = 1;
    uavs.array_size = 1;
    uavs.type = descriptor_type::texture_unordered_access_view;
    uavs.visibility = shader_stage::compute;
    const pipeline_layout_param params[3] = {pipeline_layout_param(constants), pipeline_layout_param(srvs),
                                             pipeline_layout_param(uavs)};
    if (!dev->create_pipeline_layout(3, params, &gpu.layout)) {
      gpu.failed = true;
      Log(reshade::log::level::warning, "could not create pipeline layout");
      return false;
    }
    shader_desc cs = {};
    cs.code = compute_shader.data();
    cs.code_size = compute_shader.size();
    cs.entry_point = "main";
    const pipeline_subobject sub = {pipeline_subobject_type::compute_shader, 1, &cs};
    if (!dev->create_pipeline(gpu.layout, 1, &sub, &gpu.pipe)) {
      gpu.failed = true;
      Log(reshade::log::level::warning, "could not create compute pipeline");
      return false;
    }
  }
  Log(reshade::log::level::info, "ready (" + std::to_string(width) + "x" + std::to_string(height) + ")");
  return true;
}

// Set per present by the add-on: the scan is only the fallback when the game position is unavailable.
inline std::atomic<bool> wanted = true;

// clear_depth_stencil_view: copy the scene depth-stencil before the game clears it.
inline bool OnClearDepthStencil(command_list* cmd_list, resource_view dsv, const float*, const uint8_t*, uint32_t,
                                const rect*) {
  if (!wanted.load(std::memory_order_relaxed)) return false;
  const uint64_t scene = scene_depth.load(std::memory_order_relaxed);
  if (scene == 0 || dsv.handle == 0) return false;
  auto* dev = cmd_list->get_device();
  const resource depth = dev->get_resource_from_view(dsv);
  if (depth.handle != scene) return false;

  std::scoped_lock lock(gpu_mutex);
  if (!EnsureGpu(dev, dev->get_resource_desc(depth))) return false;
  cmd_list->barrier(depth, resource_usage::depth_stencil_write, resource_usage::copy_source);
  cmd_list->copy_resource(depth, gpu.copy);
  cmd_list->barrier(depth, resource_usage::copy_source, resource_usage::depth_stencil_write);
  gpu.copy_valid = true;
  std::scoped_lock rlock(result_mutex);
  result.copies++;
  return false;
}

// Reads the output written kReadbackSlots-1 presents ago and updates the result.
inline void ConsumeReadback(uint32_t slot, float home_u, float home_v) {
  void* mapped = nullptr;
  if (!gpu.dev->map_buffer_region(gpu.readback[slot], 0, kOutputBytes, map_access::read_only, &mapped) || mapped == nullptr) {
    return;
  }
  static uint32_t data[kOutputUints];  // only used under gpu_mutex
  std::memcpy(data, mapped, sizeof(data));
  gpu.dev->unmap_buffer_region(gpu.readback[slot]);
  if (data[0] != gpu.readback_frame[slot]) return;  // not finished or stale

  const float height = static_cast<float>(gpu.height);
  const float width = static_cast<float>(gpu.width);
  const float aspect = width / height;
  const float box_x = static_cast<float>(static_cast<int32_t>(data[kOutBoxMinX]));
  const float box_y = static_cast<float>(static_cast<int32_t>(data[kOutBoxMinY]));

  // Slot -> stencil value, from the set this scan used.
  int slot_value[kSlotCount];
  uint32_t slots_used = 0;
  for (uint32_t value = 0; value < 256 && slots_used < kSlotCount; ++value) {
    if ((gpu.readback_set[slot][value >> 5] >> (value & 31)) & 1) slot_value[slots_used++] = static_cast<int>(value);
  }

  struct Group {
    uint64_t count = 0, sum_x = 0, sum_y = 0;
    uint32_t max_depth_bits = 0;
    float U(float box, float size) const { return (static_cast<float>(sum_x) / count + box + 0.5f) / size; }
    float V(float box, float size) const { return (static_cast<float>(sum_y) / count + box + 0.5f) / size; }
    void Add(const uint32_t* bucket) {
      count += bucket[0];
      sum_x += bucket[1];
      sum_y += bucket[2];
      max_depth_bits = (std::max)(max_depth_bits, bucket[3]);
    }
  };
  auto bucket_at = [](uint32_t s, int b) { return &data[kOutBuckets + (s * kBucketCount + b) * kBucketFields]; };
  auto screen_dist = [aspect](float u0, float v0, float u1, float v1) {
    const float dx = (u0 - u1) * aspect;
    const float dy = v0 - v1;
    return std::sqrt(dx * dx + dy * dy);
  };
  float last_u = 0.f, last_v = 0.f;
  bool have_last = false;
  {
    std::scoped_lock lock(result_mutex);
    have_last = result.valid;
    last_u = result.u;
    last_v = result.v;
  }

  // Candidate = one stencil value, one depth band (+ its neighbours). Pick the
  // one whose centroid is nearest to home (and to the last position).
  int best_slot = -1, best_bucket = -1;
  float best_cost = 1e30f, best_u = 0.f, best_v = 0.f;
  for (uint32_t s = 0; s < slots_used; ++s) {
    for (int b = 0; b < static_cast<int>(kBucketCount); ++b) {
      if (bucket_at(s, b)[0] < min_pixels / 4) continue;
      Group g;
      for (int n = (std::max)(0, b - 1); n <= (std::min)(static_cast<int>(kBucketCount) - 1, b + 1); ++n) g.Add(bucket_at(s, n));
      const float u = g.U(box_x, width);
      const float v = g.V(box_y, height);
      float cost = screen_dist(u, v, home_u, home_v);
      if (have_last) cost += last_weight * screen_dist(u, v, last_u, last_v);
      if (cost < best_cost) {
        best_cost = cost;
        best_slot = static_cast<int>(s);
        best_bucket = b;
        best_u = u;
        best_v = v;
      }
    }
  }
  // Merge every value's groups near the chosen one (same depth band +-1).
  Group cluster;
  uint32_t chosen[8] = {};
  if (best_slot >= 0) {
    for (uint32_t s = 0; s < slots_used; ++s) {
      for (int b = (std::max)(0, best_bucket - 1); b <= (std::min)(static_cast<int>(kBucketCount) - 1, best_bucket + 1); ++b) {
        const uint32_t* bucket = bucket_at(s, b);
        if (bucket[0] == 0) continue;
        Group g;
        g.Add(bucket);
        if (static_cast<int>(s) != best_slot
            && screen_dist(g.U(box_x, width), g.V(box_y, height), best_u, best_v) > merge_radius) {
          continue;
        }
        cluster.Add(bucket);
        chosen[slot_value[s] >> 5] |= 1u << (slot_value[s] & 31);
      }
    }
  }
  // Most common stencil values in the scanned circle.
  constexpr uint32_t kHistogram = kOutHistogram;
  uint32_t top[6][2] = {};
  for (uint32_t value = 0; value < 256; ++value) {
    const uint32_t count = data[kHistogram + value];
    for (int i = 0; i < 6; ++i) {
      if (count > top[i][1]) {
        for (int j = 5; j > i; --j) {
          top[j][0] = top[j - 1][0];
          top[j][1] = top[j - 1][1];
        }
        top[i][0] = value;
        top[i][1] = count;
        break;
      }
    }
  }

  std::scoped_lock lock(result_mutex);
  std::memcpy(result.top_stencil, top, sizeof(top));
  result.total_character_pixels = data[1];
  result.hint_stencil = data[2];
  std::memcpy(&result.hint_depth, &data[3], 4);
  result.found_now = false;
  if (cluster.count >= min_pixels) {
    float depth = 0.f;
    std::memcpy(&depth, &cluster.max_depth_bits, 4);
    const float u = cluster.U(box_x, width);
    const float v = cluster.V(box_y, height);
    const float k = result.valid ? smoothing : 1.f;
    result.u += (u - result.u) * k;
    result.v += (v - result.v) * k;
    result.depth += (depth - result.depth) * k;
    result.pixels = static_cast<uint32_t>(cluster.count);
    std::memcpy(result.chosen_values, chosen, sizeof(chosen));
    result.valid = true;
    result.found_now = true;
    result.frames_since_found = 0;
    result.last_found = std::chrono::steady_clock::now();
  }
  if (!result.found_now) result.frames_since_found++;
  if (result.valid) {
    result.seconds_since_found =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - result.last_found).count();
  }
}

// reshade_present: dispatch the scan over the latest copy and read back an older result.
inline void OnPresent(effect_runtime* runtime, float hint_u, float hint_v, float home_u, float home_v, bool enabled) {
  if (!enabled) return;
  std::scoped_lock lock(gpu_mutex);
  if (!gpu.copy_valid || gpu.pipe.handle == 0 || gpu.failed) return;

  // Leash: pull the search center back to within home_leash of home (in screen-height units).
  if (home_leash > 0.f && gpu.height > 0) {
    const float aspect = static_cast<float>(gpu.width) / static_cast<float>(gpu.height);
    const float dx = (hint_u - home_u) * aspect;
    const float dy = hint_v - home_v;
    const float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > home_leash) {
      const float k = home_leash / dist;
      hint_u = home_u + (hint_u - home_u) * k;
      hint_v = home_v + dy * k;
    }
  }

  const uint32_t frame = ++gpu.frame;
  const uint32_t slot = frame % kReadbackSlots;
  if (gpu.readback_frame[slot] != 0) ConsumeReadback(slot, home_u, home_v);

  TrackConstants constants = {};
  constants.size[0] = gpu.width;
  constants.size[1] = gpu.height;
  constants.hint_px[0] = hint_u * gpu.width;
  constants.hint_px[1] = hint_v * gpu.height;
  constants.radius_px = search_radius * gpu.height;
  constants.box_min[0] = static_cast<int32_t>(std::floor(constants.hint_px[0] - constants.radius_px));
  constants.box_min[1] = static_cast<int32_t>(std::floor(constants.hint_px[1] - constants.radius_px));
  constants.write_mask = character_write_mask.load(std::memory_order_relaxed);
  constants.frame_id = frame;
  for (int i = 0; i < 8; ++i) {
    constants.character_set[i] = character_set[i].load(std::memory_order_relaxed);
    gpu.readback_set[slot][i] = constants.character_set[i];
  }
  const uint32_t groups = static_cast<uint32_t>(std::ceil(constants.radius_px * 2.f / 8.f)) + 1;

  auto* cmd = runtime->get_command_queue()->get_immediate_command_list();
  const D3D12_CPU_DESCRIPTOR_HANDLE srvs[2] = {Descriptor(0), Descriptor(1)};
  const D3D12_CPU_DESCRIPTOR_HANDLE uavs[1] = {Descriptor(2)};
  descriptor_table_update srv_update = {};
  srv_update.count = 2;
  srv_update.type = descriptor_type::texture_shader_resource_view;
  srv_update.descriptors = srvs;
  descriptor_table_update uav_update = {};
  uav_update.count = 1;
  uav_update.type = descriptor_type::texture_unordered_access_view;
  uav_update.descriptors = uavs;

  cmd->barrier(gpu.copy, resource_usage::copy_dest, resource_usage::shader_resource);
  cmd->bind_pipeline(pipeline_stage::compute_shader, gpu.pipe);
  cmd->push_descriptors(shader_stage::compute, gpu.layout, 1, srv_update);
  cmd->push_descriptors(shader_stage::compute, gpu.layout, 2, uav_update);
  constants.mode = 0;
  cmd->push_constants(shader_stage::compute, gpu.layout, 0, 0, sizeof(constants) / 4, &constants);
  cmd->dispatch(1, 1, 1);
  cmd->barrier(gpu.output, resource_usage::unordered_access, resource_usage::unordered_access);
  constants.mode = 1;
  cmd->push_constants(shader_stage::compute, gpu.layout, 0, 0, sizeof(constants) / 4, &constants);
  cmd->dispatch(groups, groups, 1);
  cmd->barrier(gpu.output, resource_usage::unordered_access, resource_usage::copy_source);
  cmd->copy_buffer_region(gpu.output, 0, gpu.readback[slot], 0, kOutputBytes);
  cmd->barrier(gpu.output, resource_usage::copy_source, resource_usage::unordered_access);
  cmd->barrier(gpu.copy, resource_usage::shader_resource, resource_usage::copy_dest);
  gpu.readback_frame[slot] = frame;

  std::scoped_lock rlock(result_mutex);
  result.dispatches++;
  result.search_u = hint_u;
  result.search_v = hint_v;
}

inline std::string StatusText() {
  const auto r = GetResult();
  uint32_t known = 0;
  std::string values;
  for (uint32_t value = 0; value < 256; ++value) {
    if ((character_set[value >> 5].load(std::memory_order_relaxed) >> (value & 31)) & 1) {
      known++;
      values += (values.empty() ? "" : " ") + std::to_string(value);
    }
  }
  // Most common stencil values in the search area; * = counted as character.
  std::string area;
  for (const auto& entry : r.top_stencil) {
    if (entry[1] == 0) break;
    area += (area.empty() ? "" : ", ") + std::to_string(entry[0]) + (IsCharacterStencil(entry[0]) ? "*" : "") + " x"
            + std::to_string(entry[1]);
  }
  std::string chosen;
  for (uint32_t value = 0; value < 256; ++value) {
    if ((r.chosen_values[value >> 5] >> (value & 31)) & 1) chosen += (chosen.empty() ? "" : " ") + std::to_string(value);
  }
  char text[2048];
  snprintf(text, sizeof(text),
           "Tracking: %s\nPosition: %.1f%% / %.1f%%, depth %.5f, cluster %u px (stencil %s)\nSearch center: %.1f%% / %.1f%%\n"
           "Character pixels in search area: %u\n"
           "Learned character stencil values: %u (mask 0x%02X): %s\nStencil in search area: %s\n"
           "At search center: stencil %u, depth %.5f\n"
           "Depth copies: %u, scans: %u%s",
           r.found_now ? "found" : (r.valid ? "not visible, holding last position" : "not found yet"), r.u * 100.f, r.v * 100.f, r.depth,
           r.pixels, chosen.empty() ? "-" : chosen.c_str(), r.search_u * 100.f, r.search_v * 100.f, r.total_character_pixels, known, character_write_mask.load(),
           values.c_str(), area.c_str(), r.hint_stencil, r.hint_depth, r.copies, r.dispatches,
           gpu.failed ? "\n(GPU setup failed, see ReShade.log)" : "");
  return text;
}

}  // namespace tracking
