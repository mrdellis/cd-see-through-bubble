/*
 * See-Through Bubble: the game's camera on the CPU (for the Auto mode outline).
 *
 * In Auto mode the bubble shaders turn the size in metres into a size on screen
 * with SceneConstantBuffer._invViewProjRelative. To draw the same oval on the
 * overlay, the add-on reads that matrix too:
 * 1. Find root signatures whose root CBV is b19, space35 (the SceneConstantBuffer
 *    of the depth-prepass vertex shaders, which our prepass shaders also read).
 * 2. When the game binds that root CBV (push_descriptors) and the buffer lives in
 *    an upload heap (CPU-readable), copy _bufferSizeAndInvSize (offset 64) and
 *    _invViewProjRelative (offset 480). Only reads whose render size matches the
 *    scene depth count (shadow passes bind the same buffer type with their own
 *    camera). At most one capture per present, and only while the outline needs it.
 * A buffer in GPU-only memory or a binding through a descriptor table can't be
 * read this way; then the outline shows only the cross. The status text says why.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include <include/reshade.hpp>

namespace scene_camera {

using namespace reshade::api;

constexpr uint32_t kRegister = 19;
constexpr uint32_t kSpace = 35;
constexpr uint64_t kSizeOffset = 64;     // _bufferSizeAndInvSize (same offsets as bubble.hlsli)
constexpr uint64_t kMatrixOffset = 480;  // _invViewProjRelative, column_major float4x4
constexpr uint64_t kReadBytes = kMatrixOffset + 64;
constexpr uint64_t kMaxAgePresents = 30;  // older captures count as unknown
constexpr uint32_t kMaxAttemptsPerPresent = 16;

struct Camera {
  float buffer_size[4] = {};     // width, height, 1/width, 1/height
  float inv_view_proj[16] = {};  // as stored: column j at [4j .. 4j+3]
  uint64_t present = 0;          // when captured
};

inline std::atomic<bool> wanted = false;  // set per present by the add-on
// Size of the main view (the swap chain); only its aspect ratio is compared.
inline std::atomic<uint32_t> expected_width = 0;
inline std::atomic<uint32_t> expected_height = 0;
inline std::atomic<uint64_t> present_count = 1;
inline std::atomic<uint64_t> captured_present = 0;
inline std::atomic<uint32_t> attempts_this_present = 0;

inline std::shared_mutex layouts_mutex;
inline std::unordered_map<uint64_t, uint32_t> root_param_by_layout;  // layout -> root CBV param index
inline std::atomic<uint32_t> layouts_root = 0;
inline std::atomic<uint32_t> layouts_table = 0;

inline std::mutex readable_mutex;
inline std::unordered_map<uint64_t, bool> readable_by_buffer;  // buffer in an upload heap

inline std::mutex camera_mutex;
inline Camera camera;
inline bool ever_captured = false;
inline std::atomic<uint32_t> reads = 0;
inline std::atomic<uint32_t> unreadable = 0;
inline std::atomic<uint32_t> rejected = 0;
inline std::atomic<uint32_t> other_cameras = 0;

inline void Log(reshade::log::level lvl, const std::string& text) {
  reshade::log::message(lvl, ("seethroughbubble camera: " + text).c_str());
}

inline bool Covers(const descriptor_range& range) {
  return range.type == descriptor_type::constant_buffer && range.dx_register_space == kSpace
         && range.dx_register_index <= kRegister && kRegister < range.dx_register_index + (std::max)(range.count, 1u);
}

inline void OnInitPipelineLayout(device* device, uint32_t param_count, const pipeline_layout_param* params,
                                 pipeline_layout layout) {
  if (layout.handle == 0 || device->get_api() != device_api::d3d12) return;
  for (uint32_t i = 0; i < param_count; ++i) {
    const auto& param = params[i];
    if (param.type == pipeline_layout_param_type::push_descriptors && Covers(param.push_descriptors)) {
      if (layouts_root++ < 3) Log(reshade::log::level::info, "scene constants as root CBV at param " + std::to_string(i));
      std::unique_lock lock(layouts_mutex);
      root_param_by_layout[layout.handle] = i;
      return;
    }
    if (param.type == pipeline_layout_param_type::descriptor_table) {
      for (uint32_t r = 0; r < param.descriptor_table.count; ++r) {
        if (!Covers(param.descriptor_table.ranges[r])) continue;
        if (layouts_table++ < 3) {
          Log(reshade::log::level::info, "scene constants in a descriptor table at param " + std::to_string(i));
        }
        return;
      }
    }
  }
}

inline void OnDestroyPipelineLayout(device*, pipeline_layout layout) {
  std::unique_lock lock(layouts_mutex);
  root_param_by_layout.erase(layout.handle);
}

inline bool IsReadable(device* device, resource buffer) {
  {
    std::scoped_lock lock(readable_mutex);
    if (auto it = readable_by_buffer.find(buffer.handle); it != readable_by_buffer.end()) return it->second;
  }
  const auto desc = device->get_resource_desc(buffer);
  const bool readable = desc.type == resource_type::buffer && desc.heap == memory_heap::cpu_to_gpu;
  std::scoped_lock lock(readable_mutex);
  readable_by_buffer[buffer.handle] = readable;
  return readable;
}

inline bool Plausible(const Camera& c) {
  if (!(c.buffer_size[0] >= 16.f && c.buffer_size[1] >= 16.f && c.buffer_size[0] <= 16384.f
        && c.buffer_size[1] <= 16384.f)) {
    return false;
  }
  if (std::abs(c.buffer_size[2] * c.buffer_size[0] - 1.f) > 0.01f) return false;
  for (float value : c.inv_view_proj) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

inline void OnPushDescriptors(command_list* cmd_list, shader_stage, pipeline_layout layout, uint32_t layout_param,
                              const descriptor_table_update& update) {
  if (!wanted.load(std::memory_order_relaxed)) return;
  if (update.type != descriptor_type::constant_buffer || update.count == 0 || update.binding != 0
      || update.descriptors == nullptr) return;
  const uint64_t present = present_count.load(std::memory_order_relaxed);
  if (captured_present.load(std::memory_order_relaxed) == present) return;
  if (attempts_this_present.load(std::memory_order_relaxed) >= kMaxAttemptsPerPresent) return;
  {
    std::shared_lock lock(layouts_mutex);
    auto it = root_param_by_layout.find(layout.handle);
    if (it == root_param_by_layout.end() || it->second != layout_param) return;
  }
  if (attempts_this_present.fetch_add(1, std::memory_order_relaxed) >= kMaxAttemptsPerPresent) return;

  const auto& range = *static_cast<const buffer_range*>(update.descriptors);
  if (range.buffer.handle == 0) return;
  auto* device = cmd_list->get_device();
  if (!IsReadable(device, range.buffer)) {
    if (unreadable++ == 0) {
      Log(reshade::log::level::info, "scene constants are in GPU-only memory; the Auto outline shows the cross only");
    }
    return;
  }
  const auto desc = device->get_resource_desc(range.buffer);
  if (range.offset + kReadBytes > desc.buffer.size) return;

  void* mapped = nullptr;
  if (!device->map_buffer_region(range.buffer, range.offset, kReadBytes, map_access::read_only, &mapped)
      || mapped == nullptr) {
    return;
  }
  Camera captured;
  std::memcpy(captured.buffer_size, static_cast<const uint8_t*>(mapped) + kSizeOffset, sizeof(captured.buffer_size));
  std::memcpy(captured.inv_view_proj, static_cast<const uint8_t*>(mapped) + kMatrixOffset,
              sizeof(captured.inv_view_proj));
  device->unmap_buffer_region(range.buffer);
  if (!Plausible(captured)) {
    rejected++;
    return;
  }
  const uint32_t width = expected_width.load(std::memory_order_relaxed);
  const uint32_t height = expected_height.load(std::memory_order_relaxed);
  // Same aspect ratio as the scene (the size itself may be the render or the output resolution).
  if (width != 0 && height != 0) {
    const float aspect = captured.buffer_size[0] / captured.buffer_size[1];
    const float expected_aspect = static_cast<float>(width) / static_cast<float>(height);
    if (std::abs(aspect / expected_aspect - 1.f) > 0.01f) {
      if (other_cameras++ < 4) {
        char text[128];
        snprintf(text, sizeof(text), "skipped a %.0fx%.0f camera (scene %ux%u)", captured.buffer_size[0],
                 captured.buffer_size[1], width, height);
        Log(reshade::log::level::info, text);
      }
      return;  // shadow map or another pass
    }
  }
  captured.present = present;
  captured_present.store(present, std::memory_order_relaxed);
  reads++;
  std::scoped_lock lock(camera_mutex);
  camera = captured;
  if (!ever_captured) {
    ever_captured = true;
    char text[128];
    snprintf(text, sizeof(text), "first capture, render size %.0fx%.0f", captured.buffer_size[0],
             captured.buffer_size[1]);
    Log(reshade::log::level::info, text);
  }
}

inline void OnPresent() {
  present_count.fetch_add(1, std::memory_order_relaxed);
  attempts_this_present.store(0, std::memory_order_relaxed);
}

// The camera read elsewhere (player_probe's table-CBV readback), counted as captured now.
inline void Provide(const float buffer_size[2], const float inv_view_proj[16]) {
  Camera c;
  c.buffer_size[0] = buffer_size[0];
  c.buffer_size[1] = buffer_size[1];
  c.buffer_size[2] = buffer_size[0] > 0.f ? 1.f / buffer_size[0] : 0.f;
  c.buffer_size[3] = buffer_size[1] > 0.f ? 1.f / buffer_size[1] : 0.f;
  std::memcpy(c.inv_view_proj, inv_view_proj, sizeof(c.inv_view_proj));
  c.present = present_count.load(std::memory_order_relaxed);
  std::scoped_lock lock(camera_mutex);
  camera = c;
}

inline std::optional<Camera> Get() {
  std::scoped_lock lock(camera_mutex);
  if (camera.present == 0 || present_count.load(std::memory_order_relaxed) - camera.present > kMaxAgePresents) {
    return std::nullopt;
  }
  return camera;
}

// Camera-relative position of screen point (u, v) at raw depth z, as in bubble.hlsli.
inline bool Unproject(const Camera& c, float u, float v, float z, float out[3]) {
  const float ndc[4] = {u * 2.f - 1.f, 1.f - v * 2.f, z, 1.f};
  float r[4] = {};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) r[i] += c.inv_view_proj[4 * j + i] * ndc[j];
  }
  if (!(std::abs(r[3]) > 1e-20f)) return false;
  for (int i = 0; i < 3; ++i) out[i] = r[i] / r[3];
  return true;
}

// Auto mode bubble half-height as a fraction of screen height (what the shaders compute).
inline std::optional<float> AutoRadius(float u, float v, float depth, float world_radius) {
  if (!(depth > 0.f) || !(world_radius > 0.f)) return std::nullopt;
  const auto c = Get();
  if (!c.has_value()) return std::nullopt;
  float at[3], above[3];
  if (!Unproject(*c, u, v, depth, at) || !Unproject(*c, u, v + 0.01f, depth, above)) return std::nullopt;
  const float dx = above[0] - at[0], dy = above[1] - at[1], dz = above[2] - at[2];
  const float metres_per_height = std::sqrt(dx * dx + dy * dy + dz * dz) / 0.01f;
  if (!(metres_per_height > 1e-4f) || !std::isfinite(metres_per_height)) return std::nullopt;
  return world_radius / metres_per_height;
}

inline std::string StatusText() {
  if (Get().has_value()) return "Outline size: from the game camera";
  if (layouts_root.load() == 0) {
    return layouts_table.load() != 0 ? "Outline size: unavailable (scene constants in a descriptor table)"
                                     : "Outline size: unavailable (scene constants not found)";
  }
  if (unreadable.load() != 0) return "Outline size: unavailable (scene constants in GPU-only memory)";
  if (reads.load() == 0 && other_cameras.load() != 0) {
    return "Outline size: unavailable (only other cameras seen: " + std::to_string(other_cameras.load()) + ")";
  }
  if (rejected.load() != 0) return "Outline size: unavailable (" + std::to_string(rejected.load()) + " implausible reads)";
  return "Outline size: waiting for the game camera";
}

inline void Use(DWORD fdw_reason) {
  switch (fdw_reason) {
    case DLL_PROCESS_ATTACH:
      reshade::register_event<reshade::addon_event::init_pipeline_layout>(OnInitPipelineLayout);
      reshade::register_event<reshade::addon_event::destroy_pipeline_layout>(OnDestroyPipelineLayout);
      reshade::register_event<reshade::addon_event::push_descriptors>(OnPushDescriptors);
      break;
    case DLL_PROCESS_DETACH:
      reshade::unregister_event<reshade::addon_event::init_pipeline_layout>(OnInitPipelineLayout);
      reshade::unregister_event<reshade::addon_event::destroy_pipeline_layout>(OnDestroyPipelineLayout);
      reshade::unregister_event<reshade::addon_event::push_descriptors>(OnPushDescriptors);
      break;
    default: break;
  }
}

}  // namespace scene_camera
