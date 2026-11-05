/*
 * See-Through Bubble: render recon.
 * Logs every pipeline at creation and, on request, which pipelines are drawn
 * through which ReShade draw event. Used to map Crimson Desert's GPU-driven
 * passes (the DevKit snapshot misses most of its graphics draws).
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>

#include <include/reshade.hpp>

#include "../../utils/hash.hpp"
#include "./dxil.hpp"

namespace recon {

struct PipeInfo {
  // CRC32 per stage, same hash as RenoDX / DevKit.
  uint32_t vs = 0, hs = 0, ds = 0, gs = 0, ps = 0, cs = 0, as = 0, ms = 0;
  uint32_t rt_count = 0;
  uint32_t rt_formats[8] = {};
  uint32_t dsv_format = 0;
  bool has_depth_state = false;
  bool depth_enable = true;
  bool depth_write = true;
  uint32_t depth_func = static_cast<uint32_t>(reshade::api::compare_op::less);
  bool stencil_enable = false;
  uint32_t cull = static_cast<uint32_t>(reshade::api::cull_mode::back);
  std::string vs_name, ps_name;  // DXIL entry-point names
};

enum EventKind : uint32_t {
  kDraw,
  kDrawIndexed,
  kIndirectDraw,
  kIndirectDrawIndexed,
  kIndirectDispatch,
  kIndirectDispatchMesh,
  kIndirectDispatchRays,
  kIndirectUnknown,
  kDispatchMesh,
  kExecuteBundle,
  kEventKindCount,
};

constexpr const char* kEventNames[kEventKindCount] = {
    "draw",          "draw_indexed",          "indirect_draw",         "indirect_draw_indexed",
    "indirect_dispatch", "indirect_dispatch_mesh", "indirect_dispatch_rays", "indirect_unknown",
    "dispatch_mesh", "execute_bundle",
};

inline std::filesystem::path base_dir;
inline std::mutex csv_mutex;
inline std::ofstream pipelines_csv;

inline std::shared_mutex pipes_mutex;
inline std::unordered_map<uint64_t, PipeInfo> pipes;

inline std::mutex bound_mutex;
inline std::unordered_map<reshade::api::command_list*, uint64_t> bound_graphics;

// Capture state. Counted while capture_frames_left > 0.
constexpr uint32_t kCaptureFrames = 120;
inline std::atomic<uint32_t> capture_frames_left = 0;
inline std::mutex capture_mutex;
struct Agg {
  uint64_t calls = 0;
  uint64_t max_commands = 0;  // indirect: sum of max command counts
};
inline std::map<std::tuple<uint32_t, uint64_t>, Agg> capture;  // (event kind, pipeline) -> stats
inline std::atomic<uint64_t> totals[kEventKindCount] = {};

inline std::string Hex(uint32_t v) {
  char buf[16];
  snprintf(buf, sizeof(buf), "0x%08X", v);
  return buf;
}

inline uint32_t HashShader(const reshade::api::pipeline_subobject& sub) {
  const auto* desc = static_cast<const reshade::api::shader_desc*>(sub.data);
  if (desc == nullptr || desc->code == nullptr || desc->code_size == 0) return 0;
  return renodx::utils::hash::ComputeCRC32(static_cast<const uint8_t*>(desc->code), desc->code_size);
}

inline std::string EntryName(const reshade::api::pipeline_subobject& sub) {
  const auto* desc = static_cast<const reshade::api::shader_desc*>(sub.data);
  if (desc == nullptr || desc->code == nullptr || desc->code_size == 0) return {};
  return std::string(dxil::EntryPointName({static_cast<const uint8_t*>(desc->code), desc->code_size}));
}

inline PipeInfo Parse(uint32_t subobject_count, const reshade::api::pipeline_subobject* subobjects) {
  using T = reshade::api::pipeline_subobject_type;
  PipeInfo info;
  for (uint32_t i = 0; i < subobject_count; ++i) {
    const auto& sub = subobjects[i];
    if (sub.data == nullptr || sub.count == 0) continue;
    switch (sub.type) {
      case T::vertex_shader:
        info.vs = HashShader(sub);
        info.vs_name = EntryName(sub);
        break;
      case T::hull_shader:          info.hs = HashShader(sub); break;
      case T::domain_shader:        info.ds = HashShader(sub); break;
      case T::geometry_shader:      info.gs = HashShader(sub); break;
      case T::pixel_shader:
        info.ps = HashShader(sub);
        info.ps_name = EntryName(sub);
        break;
      case T::compute_shader:       info.cs = HashShader(sub); break;
      case T::amplification_shader: info.as = HashShader(sub); break;
      case T::mesh_shader:          info.ms = HashShader(sub); break;
      case T::render_target_formats: {
        const auto* formats = static_cast<const reshade::api::format*>(sub.data);
        info.rt_count = sub.count;
        for (uint32_t k = 0; k < sub.count && k < 8; ++k) info.rt_formats[k] = static_cast<uint32_t>(formats[k]);
        break;
      }
      case T::depth_stencil_format:
        info.dsv_format = static_cast<uint32_t>(*static_cast<const reshade::api::format*>(sub.data));
        break;
      case T::depth_stencil_state: {
        const auto* ds = static_cast<const reshade::api::depth_stencil_desc*>(sub.data);
        info.has_depth_state = true;
        info.depth_enable = ds->depth_enable;
        info.depth_write = ds->depth_write_mask;
        info.depth_func = static_cast<uint32_t>(ds->depth_func);
        info.stencil_enable = ds->stencil_enable;
        break;
      }
      case T::rasterizer_state:
        info.cull = static_cast<uint32_t>(static_cast<const reshade::api::rasterizer_desc*>(sub.data)->cull_mode);
        break;
      default: break;
    }
  }
  return info;
}

inline std::string Describe(const PipeInfo& p) {
  std::ostringstream s;
  s << Hex(p.vs) << ',' << Hex(p.hs) << ',' << Hex(p.ds) << ',' << Hex(p.gs) << ',' << Hex(p.ps) << ','
    << Hex(p.cs) << ',' << Hex(p.as) << ',' << Hex(p.ms) << ',' << p.rt_count << ',';
  for (uint32_t k = 0; k < p.rt_count && k < 8; ++k) s << (k ? ";" : "") << p.rt_formats[k];
  s << ',' << p.dsv_format << ',' << p.has_depth_state << ',' << p.depth_enable << ',' << p.depth_write << ','
    << p.depth_func << ',' << p.stencil_enable << ',' << p.cull << ',' << p.vs_name << ',' << p.ps_name;
  return s.str();
}

constexpr const char* kDescribeHeader =
    "vs,hs,ds,gs,ps,cs,as,ms,rt_count,rt_formats,dsv_format,has_depth_state,depth_enable,depth_write,depth_func,"
    "stencil_enable,cull,vs_name,ps_name";

inline void OnInitPipeline(reshade::api::device* device, reshade::api::pipeline_layout,
                           uint32_t subobject_count, const reshade::api::pipeline_subobject* subobjects,
                           reshade::api::pipeline pipeline) {
  if (pipeline.handle == 0u) return;
  auto info = Parse(subobject_count, subobjects);
  {
    std::unique_lock lock(pipes_mutex);
    pipes[pipeline.handle] = info;
  }
  std::scoped_lock lock(csv_mutex);
  if (pipelines_csv.is_open()) {
    pipelines_csv << Hex(static_cast<uint32_t>(pipeline.handle)) << ',' << Describe(info) << '\n';
  }
}

inline void OnBindPipeline(reshade::api::command_list* cmd_list, reshade::api::pipeline_stage stages,
                           reshade::api::pipeline pipeline) {
  if (capture_frames_left.load(std::memory_order_relaxed) == 0) return;  // free when idle
  // Compute-only binds don't change the graphics pipeline.
  if (stages == reshade::api::pipeline_stage::compute_shader) return;
  std::scoped_lock lock(bound_mutex);
  bound_graphics[cmd_list] = pipeline.handle;
}

inline void OnResetCommandList(reshade::api::command_list* cmd_list) {
  if (capture_frames_left.load(std::memory_order_relaxed) == 0) return;
  std::scoped_lock lock(bound_mutex);
  bound_graphics.erase(cmd_list);
}

inline void Record(reshade::api::command_list* cmd_list, EventKind kind, uint64_t commands = 0) {
  if (capture_frames_left.load(std::memory_order_relaxed) == 0) return;
  totals[kind].fetch_add(1, std::memory_order_relaxed);
  uint64_t pipeline = 0;
  if (cmd_list != nullptr) {
    std::scoped_lock lock(bound_mutex);
    if (auto it = bound_graphics.find(cmd_list); it != bound_graphics.end()) pipeline = it->second;
  }
  std::scoped_lock lock(capture_mutex);
  auto& agg = capture[{kind, pipeline}];
  agg.calls++;
  agg.max_commands += commands;
}

inline bool OnDraw(reshade::api::command_list* cmd_list, uint32_t, uint32_t, uint32_t, uint32_t) {
  Record(cmd_list, kDraw);
  return false;
}

inline bool OnDrawIndexed(reshade::api::command_list* cmd_list, uint32_t, uint32_t, uint32_t, int32_t, uint32_t) {
  Record(cmd_list, kDrawIndexed);
  return false;
}

inline bool OnDispatchMesh(reshade::api::command_list* cmd_list, uint32_t, uint32_t, uint32_t) {
  Record(cmd_list, kDispatchMesh);
  return false;
}

inline bool OnIndirect(reshade::api::command_list* cmd_list, reshade::api::indirect_command type,
                       reshade::api::resource, uint64_t, uint32_t draw_count, uint32_t) {
  using I = reshade::api::indirect_command;
  EventKind kind = kIndirectUnknown;
  switch (type) {
    case I::draw:          kind = kIndirectDraw; break;
    case I::draw_indexed:  kind = kIndirectDrawIndexed; break;
    case I::dispatch:      kind = kIndirectDispatch; break;
    case I::dispatch_mesh: kind = kIndirectDispatchMesh; break;
    case I::dispatch_rays: kind = kIndirectDispatchRays; break;
    default:               break;
  }
  // Plain compute dispatches are not interesting here and are very frequent.
  if (kind == kIndirectDispatch) {
    if (capture_frames_left.load(std::memory_order_relaxed) != 0) totals[kind].fetch_add(1);
    return false;
  }
  Record(cmd_list, kind, draw_count);
  return false;
}

inline void OnExecuteSecondary(reshade::api::command_list* cmd_list, reshade::api::command_list*) {
  Record(cmd_list, kExecuteBundle);
}

inline void WriteReport() {
  std::ofstream out(base_dir / "seethroughbubble-recon.txt", std::ios::trunc);
  out << "frames," << kCaptureFrames << "\n";
  for (uint32_t k = 0; k < kEventKindCount; ++k) {
    out << "total," << kEventNames[k] << ',' << totals[k].load() << '\n';
  }
  out << "event,pipeline,calls,max_commands," << kDescribeHeader << '\n';
  std::shared_lock pipes_lock(pipes_mutex);
  std::scoped_lock lock(capture_mutex);
  for (const auto& [key, agg] : capture) {
    const auto& [kind, pipeline] = key;
    out << kEventNames[kind] << ',' << Hex(static_cast<uint32_t>(pipeline)) << ',' << agg.calls << ','
        << agg.max_commands << ',';
    if (auto it = pipes.find(pipeline); it != pipes.end()) {
      out << Describe(it->second);
    } else {
      out << "unknown-pipeline";
    }
    out << '\n';
  }
}

inline std::atomic<bool> capture_requested = false;  // set by the settings button

inline void StartCapture() {
  {
    std::scoped_lock lock(capture_mutex);
    capture.clear();
  }
  {
    std::scoped_lock lock(bound_mutex);
    bound_graphics.clear();
  }
  for (auto& t : totals) t.store(0);
  capture_frames_left.store(kCaptureFrames + 1);  // first present only opens the window
  reshade::log::message(reshade::log::level::info, "seethroughbubble recon: capture started");
}

inline void OnPresent(reshade::api::effect_runtime* runtime) {
  static uint32_t frame = 0;
  ++frame;

  auto left = capture_frames_left.load();
  if (left > 0) {
    if (--left == 0) {
      capture_frames_left.store(0);
      {
        std::scoped_lock lock(csv_mutex);
        if (pipelines_csv.is_open()) pipelines_csv.flush();
      }
      WriteReport();
      reshade::log::message(reshade::log::level::info, "seethroughbubble recon: report written");
    } else {
      capture_frames_left.store(left);
    }
    return;
  }

  bool start = capture_requested.exchange(false);
  if (!start && (frame % 30) == 0) {
    std::error_code ec;
    const auto trigger = base_dir / "seethroughbubble-recon.trigger";
    if (std::filesystem::exists(trigger, ec)) {
      std::filesystem::remove(trigger, ec);
      start = true;
    }
  }
  if (start) StartCapture();
}

inline void Use(DWORD fdw_reason, HMODULE module) {
  switch (fdw_reason) {
    case DLL_PROCESS_ATTACH: {
      wchar_t path[MAX_PATH] = {};
      GetModuleFileNameW(module, path, MAX_PATH);
      base_dir = std::filesystem::path(path).parent_path();
      // Development aid: only active when seethroughbubble-recon.enable exists.
      {
        std::error_code ec;
        if (!std::filesystem::exists(base_dir / "seethroughbubble-recon.enable", ec)) break;
      }
      pipelines_csv.open(base_dir / "seethroughbubble-pipelines.csv", std::ios::trunc);
      pipelines_csv << "pipeline," << kDescribeHeader << '\n';

      reshade::register_event<reshade::addon_event::init_pipeline>(OnInitPipeline);
      reshade::register_event<reshade::addon_event::bind_pipeline>(OnBindPipeline);
      reshade::register_event<reshade::addon_event::reset_command_list>(OnResetCommandList);
      reshade::register_event<reshade::addon_event::draw>(OnDraw);
      reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
      reshade::register_event<reshade::addon_event::dispatch_mesh>(OnDispatchMesh);
      reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(OnIndirect);
      reshade::register_event<reshade::addon_event::execute_secondary_command_list>(OnExecuteSecondary);
      reshade::register_event<reshade::addon_event::reshade_present>(OnPresent);
      break;
    }
    case DLL_PROCESS_DETACH:
      reshade::unregister_event<reshade::addon_event::init_pipeline>(OnInitPipeline);
      reshade::unregister_event<reshade::addon_event::bind_pipeline>(OnBindPipeline);
      reshade::unregister_event<reshade::addon_event::reset_command_list>(OnResetCommandList);
      reshade::unregister_event<reshade::addon_event::draw>(OnDraw);
      reshade::unregister_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
      reshade::unregister_event<reshade::addon_event::dispatch_mesh>(OnDispatchMesh);
      reshade::unregister_event<reshade::addon_event::draw_or_dispatch_indirect>(OnIndirect);
      reshade::unregister_event<reshade::addon_event::execute_secondary_command_list>(OnExecuteSecondary);
      reshade::unregister_event<reshade::addon_event::reshade_present>(OnPresent);
      {
        std::scoped_lock lock(csv_mutex);
        if (pipelines_csv.is_open()) pipelines_csv.close();
      }
      break;
    default: break;
  }
}

}  // namespace recon
