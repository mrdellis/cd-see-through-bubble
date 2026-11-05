/*
 * See-Through Bubble for Crimson Desert.
 * Built on RenoDX (Copyright (C) 2024 Carlos Lopez, MIT).
 * SPDX-License-Identifier: MIT
 */

#define ImTextureID ImU64

#define DEBUG_LEVEL_0

// Developer diagnostics: status text, probe dots, probe log lines, the stencil
// isolate slider and the tracking search radius. 0 in shipped builds; set to 1
// (here or with -DSEETHROUGH_DEBUG=1) to bring them back.
#ifndef SEETHROUGH_DEBUG
#define SEETHROUGH_DEBUG 0
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>

#include <deps/imgui/imgui.h>
#include <include/reshade.hpp>

#include <embed/shaders.h>

#include "../../mods/shader.hpp"
#include "../../utils/settings.hpp"
#include "./bubble_pipelines.hpp"
#include "./player_probe.hpp"
#include "./config_store.hpp"
#include "./recon.hpp"
#include "./scene_camera.hpp"
#include "./shared.h"
#include "./version.h"

namespace {

// Phase 0 scaffold: no game shaders are replaced yet. G-buffer pixel shaders
// are registered here once the DevKit has identified their hashes.
renodx::mods::shader::CustomShaders custom_shaders = {};

ShaderInjectData shader_injection = {
    .camera_near = 0.1f,  // fallback shaders only
    .render_width = 1920.f,
    .render_height = 1080.f,
    .bubble_width_scale = 1.f,
};

// Placement: manual values from the settings; the shader values are set per frame.
float placement_mode = 0.f;  // 0 = auto (track character), 1 = manual
float manual_center_u = 0.54f;
float manual_center_v = 0.53f;
float manual_cut_distance = 6.f;
float manual_width = 0.17f;  // fraction of screen height (bubble half-width)
float world_width = 1.3f;    // metres (bubble half-width around the character)
std::atomic<bool> manual_edit_requested = false;
std::atomic<bool> copy_tracked_requested = false;

// Auto mode source: the game's own player position (player_probe) once it has
// produced a reading; the stencil scan (tracking.hpp) only as a fallback.
struct AutoPlacement {
  bool found = false;  // usable this frame
  float u = 0.5f, v = 0.5f, depth = 0.f;
  const char* source = "none";
};
AutoPlacement auto_placement;  // last frame's, for the status text and "Use Tracked Position"
std::chrono::steady_clock::time_point probe_last_good = {};
// Readings are normally 2-4 presents old, more with frame generation (each generated
// frame is a present): accept anything the readback ring can hold.
constexpr uint64_t kProbeFreshPresents = player_probe::kSlots - 1;

// What the bubble cuts ("See Through" section); read per bind/draw by bubble_pipelines.
float cut_vegetation = 1.f;       // trees, bushes, plants (SpeedTree Leaf/Branch/Frond shaders)
float cut_objects = 1.f;          // buildings, fences (psStandard, stencil ref other than 129)
float cut_props = 1.f;            // jars and other loose props (psStandard, stencil ref 129)
float cut_terrain_meshes = 1.f;   // rocks, cliffs (psTerrainMesh and its depth prepass)
float debug_stencil_only = -1.f;  // debug: cut only world draws with this stencil reference (-1 = all)
float bubble_enabled_setting = 1.f;  // the user's on/off; the shader flag also drops when the character is lost
float toggle_key = 8.f;              // index into kToggleKeys (8 = F9)
float forget_after_seconds = 2.f;
float disable_in_cutscenes = 1.f;   // pause while the cinematic letterbox is drawn
float disable_without_hud = 0.f;    // pause while the HUD minimap isn't drawn (photo mode, menus)
std::string pause_reason;           // shown in the status text
float show_outline = 0.f;           // debug: draw the outline every frame, also with the overlay closed
float hide_outline = 0.f;           // never draw the outline (wins over show_outline)
std::atomic<uint64_t> settings_drawn_present = 0;  // last present the settings tab drew (outline wanted there too)
std::atomic<uint64_t> present_number = 0;
constexpr uint64_t kMarkerHoldPresents = 30;    // auto mode: stop cutting when the character hasn't been seen for this long
constexpr int kToggleKeys[] = {0, VK_F2, VK_F3, VK_F4, VK_F5, VK_F6, VK_F7, VK_F8, VK_F9, VK_F11, VK_F12};

// Draws the bubble's edges on top of the game while the settings tab is open, or
// every frame with "Always Show Outline": solid line where the full cut ends
// (before the dithered taper), faint line at the outer edge. Auto mode adds a
// cross at the tracked position; its oval size comes from the game camera
// (scene_camera.hpp), and without it only the cross is drawn. Grey = not cutting.
void DrawBubbleOutline();
inline float manual_height_fraction() { return shader_injection.bubble_radius; }

renodx::utils::settings::Settings settings = {
    new renodx::utils::settings::Setting{
        .key = "BubbleEnabled",
        .binding = &bubble_enabled_setting,
        .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
        .default_value = 1.f,
        .label = "Enabled",
        .section = "Bubble",
    },
    new renodx::utils::settings::Setting{
        .key = "ToggleKey",
        .binding = &toggle_key,
        .value_type = renodx::utils::settings::SettingValueType::INTEGER,
        .default_value = 8.f,
        .label = "On/Off Hotkey",
        .section = "Bubble",
        .tooltip = "Key that switches the bubble on and off in game.",
        .labels = {"None", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F11", "F12"},
    },
    new renodx::utils::settings::Setting{
        .key = "DisableInCutscenes",
        .binding = &disable_in_cutscenes,
        .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
        .default_value = 1.f,
        .label = "Disable In Cutscenes",
        .section = "Bubble",
        .tooltip = "Pause the bubble while cinematic letterbox bars are shown.",
    },
    new renodx::utils::settings::Setting{
        .key = "DisableWithoutHud",
        .binding = &disable_without_hud,
        .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
        .default_value = 0.f,
        .label = "Disable When HUD Is Hidden",
        .section = "Bubble",
        .tooltip = "Pause the bubble while the minimap isn't shown (photo mode, menus). "
                   "Leave off if you play with the minimap disabled: the bubble would never show.",
    },
    new renodx::utils::settings::Setting{
        .key = "BubbleRadius",
        .binding = &shader_injection.bubble_radius,
        .default_value = 21.f,
        .label = "Height (Manual)",
        .section = "Bubble",
        .tooltip = "Manual mode: half the bubble's height, as a percentage of screen height.",
        .min = 1.f,
        .max = 50.f,
        .parse = [](float value) { return value * 0.01f; },
    },
    new renodx::utils::settings::Setting{
        .key = "BubbleWidthManual",
        .binding = &manual_width,
        .default_value = 17.f,
        .label = "Width (Manual)",
        .section = "Bubble",
        .tooltip = "Manual mode: half the bubble's width, as a percentage of screen height. Equal to Height = circle.",
        .min = 1.f,
        .max = 50.f,
        .parse = [](float value) { return value * 0.01f; },
    },
    new renodx::utils::settings::Setting{
        .key = "BubbleWorldRadius",
        .binding = &shader_injection.bubble_world_radius,
        .default_value = 1.5f,
        .label = "Height (Auto)",
        .section = "Bubble",
        .tooltip = "Auto mode: half the bubble's height in metres around the character; it keeps its size when the camera zooms.",
        .min = 0.3f,
        .max = 4.f,
        .format = "%.1f m",
    },
    new renodx::utils::settings::Setting{
        .key = "BubbleWorldWidth",
        .binding = &world_width,
        .default_value = 1.3f,
        .label = "Width (Auto)",
        .section = "Bubble",
        .tooltip = "Auto mode: half the bubble's width in metres. Equal to Height = circle.",
        .min = 0.3f,
        .max = 4.f,
        .format = "%.1f m",
    },
    new renodx::utils::settings::Setting{
        .key = "BubbleSoftness",
        .binding = &shader_injection.bubble_softness,
        .default_value = 29.f,
        .label = "Edge Softness",
        .section = "Bubble",
        .tooltip = "Width of the dithered edge as a percentage of the radius.",
        .min = 1.f,
        .max = 100.f,
        .parse = [](float value) { return value * 0.01f; },
    },
    new renodx::utils::settings::Setting{
        .key = "CutMargin",
        .binding = &shader_injection.cut_margin,
        .default_value = 1.f,
        .label = "Cut Margin",
        .section = "Bubble",
        .tooltip = "Auto mode: cut geometry closer than the character minus this many metres.",
        .min = 0.f,
        .max = 5.f,
        .format = "%.1f m",
    },
    new renodx::utils::settings::Setting{
        .key = "CutVegetation",
        .binding = &cut_vegetation,
        .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
        .default_value = 1.f,
        .label = "Trees & Plants",
        .section = "See Through",
        .tooltip = "Cut trees, bushes and other plants in front of the character.",
    },
    new renodx::utils::settings::Setting{
        .key = "CutObjects",
        .binding = &cut_objects,
        .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
        .default_value = 1.f,
        .label = "Buildings & Structures",
        .section = "See Through",
        .tooltip = "Cut buildings, walls, fences and other structures in front of the character.",
    },
    new renodx::utils::settings::Setting{
        .key = "CutProps",
        .binding = &cut_props,
        .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
        .default_value = 1.f,
        .label = "Small Objects",
        .section = "See Through",
        .tooltip = "Cut jars and other loose objects in front of the character.",
    },
    new renodx::utils::settings::Setting{
        .key = "CutTerrainMeshes",
        .binding = &cut_terrain_meshes,
        .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
        .default_value = 1.f,
        .label = "Rocks & Cliffs",
        .section = "See Through",
        .tooltip = "Cut rocks, cliffs and other rock formations in front of the character.",
    },
    new renodx::utils::settings::Setting{
        .key = "PlacementMode",
        .binding = &placement_mode,
        .value_type = renodx::utils::settings::SettingValueType::INTEGER,
        .default_value = 0.f,
        .label = "Placement Mode",
        .section = "Placement",
        .tooltip = "Auto follows the character. Changing Center X/Y or Cut Distance switches to Manual.",
        .labels = {"Auto (track character)", "Manual"},
    },
    new renodx::utils::settings::Setting{
        .key = "ShowOutline",
        .binding = &show_outline,
        .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
        .default_value = 0.f,
        .label = "Always Show Outline (Debug)",
        .section = "Placement",
        .tooltip = "Draws the bubble outline all the time, also with the overlay closed, to check the tracking. "
                   "Yellow: cutting. Grey: not cutting (character not found, paused or off).",
    },
    new renodx::utils::settings::Setting{
        .key = "HideOutline",
        .binding = &hide_outline,
        .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
        .default_value = 0.f,
        .label = "Never Show Outline (even in Debug)",
        .section = "Placement",
        .tooltip = "Never draw the bubble outline, also while the ReShade overlay is open. "
                   "Overrides Always Show Outline.",
    },
    new renodx::utils::settings::Setting{
        .value_type = renodx::utils::settings::SettingValueType::CUSTOM,
        .section = "Placement",
        .on_draw = [] {
          if (bubble::scene_layout_changed) {
            ImGui::TextUnformatted("Disabled: the game's scene data changed (game update?).\n"
                                   "The add-on needs an update. Details in ReShade.log.");
          }
          if (!pause_reason.empty()) ImGui::TextUnformatted(("Paused: " + pause_reason).c_str());
          ImGui::TextDisabled("Version " SEETHROUGH_VERSION);
          settings_drawn_present = present_number.load();
          if (show_outline == 0.f) DrawBubbleOutline();  // otherwise drawn every frame (OnReshadeOverlay)
#if SEETHROUGH_DEBUG
          if (placement_mode == 0.f) {
            char line[160];
            std::snprintf(line, sizeof(line), "Auto source: %s, %s (%.1f%% / %.1f%%)", auto_placement.source,
                          auto_placement.found ? "following" : "not found", auto_placement.u * 100.f,
                          auto_placement.v * 100.f);
            ImGui::TextUnformatted(line);
          }
          ImGui::TextUnformatted(tracking::StatusText().c_str());
          ImGui::TextUnformatted(player_probe::StatusText().c_str());
          if (placement_mode == 0.f) ImGui::TextUnformatted(scene_camera::StatusText().c_str());
#endif
          return false;
        },
    },
    new renodx::utils::settings::Setting{
        .value_type = renodx::utils::settings::SettingValueType::BUTTON,
        .label = "Use Tracked Position As Manual",
        .section = "Placement",
        .tooltip = "Copies the tracked screen position into Center X/Y and switches to Manual.",
        .on_click = [] {
          copy_tracked_requested = true;
          return false;
        },
    },
    new renodx::utils::settings::Setting{
        .key = "BubbleCenterX",
        .binding = &manual_center_u,
        .default_value = 54.f,
        .label = "Center X",
        .section = "Placement",
        .tooltip = "Manual bubble position (percent of screen width).",
        .parse = [](float value) { return value * 0.01f; },
        .on_change = [] { manual_edit_requested = true; },
    },
    new renodx::utils::settings::Setting{
        .key = "BubbleCenterY",
        .binding = &manual_center_v,
        .default_value = 53.f,
        .label = "Center Y",
        .section = "Placement",
        .tooltip = "Manual bubble position (percent of screen height).",
        .parse = [](float value) { return value * 0.01f; },
        .on_change = [] { manual_edit_requested = true; },
    },
    new renodx::utils::settings::Setting{
        .key = "BubbleCutDistance",
        .binding = &manual_cut_distance,
        .default_value = 6.f,
        .label = "Cut Distance",
        .section = "Placement",
        .tooltip = "Manual mode: only geometry closer to the camera than this many metres is cut. "
                   "Set it a little below the camera-to-character distance.",
        .min = 0.5f,
        .max = 30.f,
        .format = "%.1f m",
        .on_change = [] { manual_edit_requested = true; },
    },
    new renodx::utils::settings::Setting{
        .key = "ForgetAfter",
        .binding = &forget_after_seconds,
        .default_value = 2.f,
        .label = "Forget Character After",
        .section = "Placement",
        .tooltip = "Auto mode: stop cutting when the character hasn't been found for this many seconds "
                   "(cutscenes, camera far away). 0 = never.",
        .min = 0.f,
        .max = 10.f,
        .format = "%.1f s",
    },
#if SEETHROUGH_DEBUG
    new renodx::utils::settings::Setting{
        .key = "DebugStencilPick",
        .binding = &debug_stencil_only,
        .value_type = renodx::utils::settings::SettingValueType::INTEGER,
        .default_value = 0.f,
        .label = "Only Cut Stencil Value (Debug)",
        .section = "Debug",
        .tooltip = "Cut only world draws whose stencil reference is this value. "
                   "To find which objects share a stencil value (values from the stencil survey in ReShade.log).",
        .labels = {"All", "1", "5", "33", "36", "96", "129", "133", "161", "224"},
        .parse = [](float value) {
          static constexpr float kValues[] = {-1.f, 1.f, 5.f, 33.f, 36.f, 96.f, 129.f, 133.f, 161.f, 224.f};
          const int i = static_cast<int>(value);
          return (i >= 0 && i < 10) ? kValues[i] : -1.f;
        },
    },
    new renodx::utils::settings::Setting{
        .key = "SearchRadius",
        .binding = &tracking::search_radius,
        .default_value = 22.f,
        .label = "Tracking Search Radius",
        .section = "Placement",
        .tooltip = "Auto mode: how far from the last position (percent of screen height) to look for the character.",
        .min = 5.f,
        .max = 50.f,
        .parse = [](float value) { return value * 0.01f; },
    },
#endif
};

bool initialized = false;

// Oval from our own points: ImDrawList::AddEllipse goes through ReShade's ImGui
// function table, and the running ReShade drew it as a circle (width ignored).
void AddOval(ImDrawList* draw_list, ImVec2 center, float rx, float ry, ImU32 color, float thickness) {
  constexpr int kSegments = 96;
  ImVec2 points[kSegments];
  for (int i = 0; i < kSegments; ++i) {
    const float angle = 6.2831853f * static_cast<float>(i) / kSegments;
    points[i] = ImVec2(center.x + std::cos(angle) * rx, center.y + std::sin(angle) * ry);
  }
  draw_list->AddPolyline(points, kSegments, color, ImDrawFlags_Closed, thickness);
}

#if SEETHROUGH_DEBUG
// Probe dots: where the game's player constants project to (see player_probe.hpp).
void DrawProbeMarkers(ImDrawList* draw_list, ImVec2 display) {
  const auto probe = player_probe::GetResult();
  if (!probe.valid) return;
  const struct {
    const player_probe::Projection* p;
    ImU32 color;
    float radius;
  } dots[] = {{&probe.head_abs, IM_COL32(255, 0, 255, 255), 9.f},
              {&probe.head_rel, IM_COL32(0, 255, 255, 255), 6.f},
              {&probe.main_abs, IM_COL32(255, 128, 192, 255), 9.f},
              {&probe.main_rel, IM_COL32(64, 96, 255, 255), 6.f}};
  for (const auto& dot : dots) {
    if (!dot.p->valid) continue;
    const ImVec2 at(dot.p->u * display.x, dot.p->v * display.y);
    draw_list->AddCircleFilled(at, dot.radius, dot.color);
    draw_list->AddCircle(at, dot.radius + 1.f, IM_COL32(0, 0, 0, 200), 0, 1.5f);
  }
}
#endif

void DrawBubbleOutline() {
  if (hide_outline != 0.f) return;
  const ImVec2 display = ImGui::GetIO().DisplaySize;
  if (display.x <= 0.f || display.y <= 0.f) return;
  auto* draw_list = ImGui::GetForegroundDrawList(nullptr);
#if SEETHROUGH_DEBUG
  DrawProbeMarkers(draw_list, display);
#endif
  const ImVec2 center(shader_injection.bubble_center_u * display.x, shader_injection.bubble_center_v * display.y);
  const bool cutting = shader_injection.bubble_enabled != 0.f;
  const ImU32 edge_color = cutting ? IM_COL32(255, 220, 64, 240) : IM_COL32(170, 170, 170, 200);
  const ImU32 outer_color = cutting ? IM_COL32(255, 255, 255, 110) : IM_COL32(170, 170, 170, 90);
  float ry = 0.f;
  if (placement_mode == 0.f) {
    const float arm = display.y * 0.02f;
    draw_list->AddLine(ImVec2(center.x - arm, center.y), ImVec2(center.x + arm, center.y), edge_color, 2.f);
    draw_list->AddLine(ImVec2(center.x, center.y - arm), ImVec2(center.x, center.y + arm), edge_color, 2.f);
    const auto radius = scene_camera::AutoRadius(shader_injection.bubble_center_u, shader_injection.bubble_center_v,
                                                 shader_injection.player_depth, shader_injection.bubble_world_radius);
    if (!radius.has_value()) return;  // not tracking, or camera unknown: cross only
    ry = *radius * display.y;
  } else {
    ry = manual_height_fraction() * display.y;
  }
  const float rx = ry * shader_injection.bubble_width_scale;  // width / height, set per present
  const float inner = (std::max)(0.f, 1.f - shader_injection.bubble_softness);
  AddOval(draw_list, center, rx, ry, outer_color, 1.5f);
  AddOval(draw_list, center, rx * inner, ry * inner, edge_color, 2.5f);
}

// Every overlay frame (also with the overlay closed): the debug outline.
void OnReshadeOverlay(reshade::api::effect_runtime*) {
  if (show_outline != 0.f) DrawBubbleOutline();
}

// Once per frame (outside the settings UI): apply placement, run tracking.
void OnReshadePresent(reshade::api::effect_runtime* runtime) {
  if (copy_tracked_requested.exchange(false)) {
    if (auto_placement.found) {
      renodx::utils::settings::UpdateSettings({{"BubbleCenterX", auto_placement.u * 100.f},
                                               {"BubbleCenterY", auto_placement.v * 100.f},
                                               {"PlacementMode", 1.f}});
      renodx::utils::settings::SaveSettings();
    }
  }
  if (manual_edit_requested.exchange(false) && placement_mode == 0.f) {
    renodx::utils::settings::UpdateSetting("PlacementMode", 1.f);
    renodx::utils::settings::SaveSettings();
  }

  // Hotkey: flip and save the user's on/off.
  const int key_index = static_cast<int>(toggle_key);
  if (key_index > 0 && key_index < static_cast<int>(std::size(kToggleKeys))
      && runtime->is_key_pressed(kToggleKeys[key_index])) {
    renodx::utils::settings::UpdateSetting("BubbleEnabled", bubble_enabled_setting != 0.f ? 0.f : 1.f);
    renodx::utils::settings::SaveSettings();
  }

  // Pause in cutscenes (letterbox drawn recently) and, optionally, while the HUD is hidden.
  const bool in_cutscene =
      disable_in_cutscenes != 0.f && bubble::PresentsSinceMarker(bubble::kMarkerLetterbox) < kMarkerHoldPresents;
  const uint64_t since_minimap = bubble::PresentsSinceMarker(bubble::kMarkerMinimap);
  const bool hud_hidden = disable_without_hud != 0.f && since_minimap != UINT64_MAX && since_minimap > kMarkerHoldPresents;
  pause_reason = in_cutscene ? "cutscene" : (hud_hidden ? "HUD hidden" : "");
  const bool paused = in_cutscene || hud_hidden;

  const bool auto_mode = placement_mode == 0.f && bubble_enabled_setting != 0.f && !paused;
  // Auto source: the game's player position, projected (see player_probe.hpp). It
  // keeps the last good reading for "Forget After" seconds while passes bind
  // other data or the character is off screen / behind the camera.
  const auto probe = player_probe::GetResult();
  const auto now = std::chrono::steady_clock::now();
  const bool probe_source = probe.valid;  // the probe has worked at least once: don't fall back
  if (probe_source && player_probe::PresentsSince(probe) <= kProbeFreshPresents
      && player_probe::PresentsSincePosition(probe) <= kProbeFreshPresents) {
    probe_last_good = now;
  }
  const float probe_age = std::chrono::duration<float>(now - probe_last_good).count();
  const bool probe_lost = forget_after_seconds > 0.f ? probe_age > forget_after_seconds : probe_age > 0.5f;
  AutoPlacement placed;
  if (probe_source) {
    placed.source = "game position";
    placed.found = probe.body.valid && !probe_lost && probe.body.depth > 0.f && probe.body.depth < 1.f;
    placed.u = probe.body.u;
    placed.v = probe.body.v;
    placed.depth = probe.body.depth;
    if (!probe_lost) scene_camera::Provide(probe.buffer_size, probe.inv_view_proj_relative);
  } else {
    const auto tracked = tracking::GetResult();
    const bool lost = tracked.valid && forget_after_seconds > 0.f && tracked.seconds_since_found > forget_after_seconds;
    placed.source = "stencil scan";
    placed.found = tracked.valid && !lost;
    placed.u = tracked.u;
    placed.v = tracked.v;
    placed.depth = tracked.depth;
  }
  auto_placement = placed;
  const bool use_tracked = auto_mode && placed.found;
  // Auto mode cuts nothing until the character has been found, and again once it is lost.
  shader_injection.bubble_enabled =
      (bubble_enabled_setting != 0.f && !paused && (placement_mode != 0.f || use_tracked)) ? 1.f : 0.f;
  shader_injection.bubble_center_u = use_tracked ? placed.u : manual_center_u;
  shader_injection.bubble_center_v = use_tracked ? placed.v : manual_center_v;
  shader_injection.player_depth = use_tracked ? placed.depth : 0.f;
  // The shaders project the (predicted) world position with their own frame's camera;
  // the screen values above remain for the fallback shaders and the outline.
  float predicted[3] = {};
  const bool use_world = use_tracked && probe_source;
  if (use_world) player_probe::PredictBody(probe, predicted);
  shader_injection.player_world_valid = use_world ? 1.f : 0.f;
  shader_injection.player_world_x = predicted[0];
  shader_injection.player_world_y = predicted[1];
  shader_injection.player_world_z = predicted[2];
  shader_injection.bubble_cut_distance = manual_cut_distance;
  const float height = placement_mode != 0.f ? shader_injection.bubble_radius : shader_injection.bubble_world_radius;
  const float width = placement_mode != 0.f ? manual_width : world_width;
  shader_injection.bubble_width_scale = height > 0.f ? (std::max)(width / height, 0.05f) : 1.f;

  tracking::home_leash = tracking::search_radius;
  tracking::wanted = auto_mode && !probe_source;
  tracking::OnPresent(runtime, use_tracked ? placed.u : manual_center_u, use_tracked ? placed.v : manual_center_v,
                      manual_center_u, manual_center_v, auto_mode && !probe_source);

  // Game camera for the Auto mode outline size, read only while an outline is shown.
  const uint64_t present = present_number.fetch_add(1) + 1;
  const bool outline_shown =
      hide_outline == 0.f && (show_outline != 0.f || present - settings_drawn_present.load() < 5);
  scene_camera::wanted = placement_mode == 0.f && outline_shown;
  uint32_t view_width = 0, view_height = 0;
  runtime->get_screenshot_width_and_height(&view_width, &view_height);
  scene_camera::expected_width = view_width;
  scene_camera::expected_height = view_height;
  scene_camera::OnPresent();

  // Probe: the player's position from the game's constants.
  player_probe::OnPresent(view_width, view_height);
  if (present % 600 == 0) {
    reshade::log::message(reshade::log::level::info, ("seethroughbubble: " + player_probe::TimingText()).c_str());
  }
#if SEETHROUGH_DEBUG
  if (present % 600 == 0) {
    std::string text = player_probe::StatusText();
    for (auto& c : text) {
      if (c == '\n') c = '|';
    }
    reshade::log::message(reshade::log::level::info, ("seethroughbubble probe: " + text).c_str());
  }
  if (present == 600 || present == 6000) {
    reshade::log::message(reshade::log::level::info, ("seethroughbubble probe: " + player_probe::SurveyText()).c_str());
  }
#endif
}

// RenoDX's shader mod inserts an injection root parameter into *every* pipeline
// layout, even with no shaders registered. Keep it off until Phase 2 needs it,
// so the recon build stays passive.
constexpr bool kUseShaderMod = false;

}  // namespace

extern "C" __declspec(dllexport) constexpr const char* NAME = "See-Through Bubble";
extern "C" __declspec(dllexport) constexpr const char* DESCRIPTION =
    "v" SEETHROUGH_VERSION ": cuts a dithered hole through geometry between the camera and the player (Crimson Desert)";

// ReShade docks add-on windows only when it first creates its layout, and it
// saves the layout only on clean shutdown or when its own settings change, which
// Crimson Desert rarely triggers. So if the saved layout has no entry for our
// window yet, add one that docks it next to ReShade's Home tab.
void EnsureOverlayDocked() {
  size_t size = 0;
  if (!reshade::get_config_value(nullptr, "OVERLAY", "Window", nullptr, &size) || size == 0) return;
  std::string buffer(size, '\0');
  if (!reshade::get_config_value(nullptr, "OVERLAY", "Window", buffer.data(), &size)) return;
  buffer.resize(size);

  std::vector<std::string> elements;
  for (size_t start = 0; start < buffer.size();) {
    size_t end = buffer.find('\0', start);
    if (end == std::string::npos) end = buffer.size();
    if (end > start) elements.emplace_back(buffer.substr(start, end - start));
    start = end + 1;
  }

  const std::string ours = std::string("[Window][") + NAME + "]";
  std::string home_dock;
  std::vector<std::pair<std::string, int>> docks;  // (dock id, tab order)
  std::string current;
  for (const auto& element : elements) {
    if (element == ours) return;  // already has saved settings (docked or deliberately floating)
    if (element.rfind("[Window]", 0) == 0) {
      current = element;
    } else if (element.rfind("DockId=", 0) == 0) {
      std::string value = element.substr(7);
      const size_t comma = value.find(',');
      const std::string id = value.substr(0, comma);
      const int order = comma == std::string::npos ? 0 : std::atoi(value.c_str() + comma + 1);
      docks.emplace_back(id, order);
      if (current == "[Window][###home]") home_dock = id;
    }
  }
  if (home_dock.empty()) return;
  int max_order = -1;
  for (const auto& [id, order] : docks) {
    if (id == home_dock) max_order = (std::max)(max_order, order);
  }

  elements.push_back(ours);
  elements.emplace_back("Collapsed=0");
  elements.push_back("DockId=" + home_dock + "," + std::to_string(max_order + 1));
  std::string joined;
  for (const auto& element : elements) {
    joined += element;
    joined.push_back('\0');
  }
  reshade::set_config_value(nullptr, "OVERLAY", "Window", joined.data(), joined.size());
  reshade::log::message(reshade::log::level::info, "seethroughbubble: docked settings window next to ReShade's Home tab");
}

// Settings live in SeeThroughBubble.ini next to the add-on. On first run, the
// values RenoDX stored in ReShade.ini are copied over once.
void UseOwnSettingsFile(HMODULE module) {
  wchar_t path[MAX_PATH] = {};
  GetModuleFileNameW(module, path, MAX_PATH);
  const auto file = std::filesystem::path(path).parent_path() / L"SeeThroughBubble.ini";
  if (!config_store::Load(file)) {
    const std::string section = "seethroughbubble-preset1";
    for (const auto* setting : settings) {
      if (setting->key.empty()) continue;
      char value[256] = {};
      size_t size = sizeof(value);
      if (reshade::get_config_value(nullptr, section.c_str(), setting->key.c_str(), value, &size) && value[0] != '\0') {
        config_store::sections[section][setting->key] = value;
      }
    }
    std::scoped_lock lock(config_store::mutex);
    config_store::SaveLocked();
  }
  renodx::utils::settings::config_reader = config_store::Read;
  renodx::utils::settings::config_writer = config_store::Write;
}

BOOL APIENTRY DllMain(HMODULE h_module, DWORD fdw_reason, LPVOID lpv_reserved) {
  switch (fdw_reason) {
    case DLL_PROCESS_ATTACH:
      if (!reshade::register_addon(h_module)) return FALSE;
      reshade::log::message(reshade::log::level::info, "seethroughbubble: version " SEETHROUGH_VERSION);
      EnsureOverlayDocked();
      UseOwnSettingsFile(h_module);

      if (!initialized) {
        renodx::mods::shader::force_pipeline_cloning = true;
        renodx::mods::shader::expected_constant_buffer_space = 50;
        renodx::mods::shader::expected_constant_buffer_index = 13;
        renodx::mods::shader::allow_multiple_push_constants = true;
        renodx::utils::settings::use_presets = false;
        renodx::utils::settings::global_name = "seethroughbubble";
        initialized = true;
      }
      break;
    case DLL_PROCESS_DETACH:
      reshade::unregister_addon(h_module);
      break;
  }

  renodx::utils::settings::Use(fdw_reason, &settings);
  if constexpr (kUseShaderMod) {
    renodx::mods::shader::Use(fdw_reason, custom_shaders, &shader_injection);
  }
  if (fdw_reason == DLL_PROCESS_ATTACH) {
    bubble::injection = &shader_injection;
    bubble::prepass_shader = __bubble_prepass;
    bubble::prepass_terrain_shader = __bubble_prepass_terrain;
    bubble::prepass_basic_shader = __bubble_prepass_basic;
    bubble::prepass_terrain_basic_shader = __bubble_prepass_terrain_basic;
    bubble::cut_vegetation = &cut_vegetation;
    bubble::cut_objects = &cut_objects;
    bubble::cut_terrain_meshes = &cut_terrain_meshes;
    bubble::cut_props = &cut_props;
    bubble::stencil_only = &debug_stencil_only;
    bubble::snippet_shader = __bubble_snippet;
  }
  if (fdw_reason == DLL_PROCESS_ATTACH) {
    // Kill switch: create bin64/seethroughbubble-disable-layouts to start the game
    // without root-signature changes (the bubble then stays inactive).
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(h_module, path, MAX_PATH);
    std::error_code ec;
    bubble::disable_layout_injection = std::filesystem::exists(
        std::filesystem::path(path).parent_path() / "seethroughbubble-disable-layouts", ec);
    bubble::cache_dir = std::filesystem::path(path).parent_path() / "seethroughbubble-cache";
  }
  if (fdw_reason == DLL_PROCESS_ATTACH) {
    tracking::compute_shader = __bubble_track;
    reshade::register_event<reshade::addon_event::reshade_present>(OnReshadePresent);
    reshade::register_event<reshade::addon_event::reshade_overlay>(OnReshadeOverlay);
  } else if (fdw_reason == DLL_PROCESS_DETACH) {
    reshade::unregister_event<reshade::addon_event::reshade_present>(OnReshadePresent);
    reshade::unregister_event<reshade::addon_event::reshade_overlay>(OnReshadeOverlay);
  }
  bubble::Use(fdw_reason);
  scene_camera::Use(fdw_reason);
  player_probe::Use(fdw_reason);
  recon::Use(fdw_reason, h_module);

  return TRUE;
}
