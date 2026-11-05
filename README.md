# See-Through Bubble for Crimson Desert

A ReShade add-on that cuts a soft, dithered hole through whatever is between the camera and your character, so trees, rocks and buildings stop blocking your view. It follows the character on its own, and you can choose what gets cut.

It's built on [RenoDX](https://github.com/clshortfuse/renodx). This repo only has the add-on source plus one small patch for RenoDX. There are no game files or dumped shaders in here. The add-on patches the game's shaders in memory as they load, using the compiler that ships with the game.

Last tested on game version 2.04.06 with ReShade 6.8.0. It works on both NVIDIA (RTX 4080) and AMD (RX 9070 XT).

## What you need

- Visual Studio 2022 with the "Desktop development with C++" workload. In the installer, also tick "C++ Clang tools for Windows", because the build uses clang-cl. Leave "C++ CMake tools for Windows" ticked too, that gets you CMake and Ninja.
- Windows SDK 10.0.26100 or newer
- Git

## Building

Open the **x64 Native Tools Command Prompt for VS 2022** and cd into this repo. It has to be that prompt, a normal one won't find cmake or clang-cl.

First get RenoDX. Stick to this commit, it's the one the patch was made against and newer ones may not work:

```
git clone https://github.com/clshortfuse/renodx.git external\renodx
cd external\renodx
git checkout 43c067b
git submodule update --init --recursive
```

Go make a coffee, the submodules take a while.

Now apply the patch. The add-on won't compile without it. It lets the add-on keep its settings in its own `SeeThroughBubble.ini` instead of `ReShade.ini`, and draws on/off settings as checkboxes.

```
git apply ..\..\patches\renodx-settings.patch
```

RenoDX doesn't keep its shader compilers in git, so download them with its setup script:

```
powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev-env.ps1 -Install
```

RenoDX builds whatever is in `src\games`, so link the add-on in there:

```
mklink /J src\games\seethroughbubble ..\..\src
```

Then build it:

```
cmake --preset clang-x64
cmake --build --preset clang-x64-release --target seethroughbubble
```

You'll find the add-on at `external\renodx\build\Release\renodx-seethroughbubble.addon64`.

After that you only need the last command. The link means your edits in `src` get picked up straight away.

## Installing

1. Install ReShade with add-on support for `bin64\CrimsonDesert.exe`. It has to be the "with add-on support" version from reshade.me. You can skip the effect packages.
2. Close the game and copy `renodx-seethroughbubble.addon64` into `bin64`, next to the exe.
3. Start the game. The first start takes a bit longer because the add-on prepares its shaders and caches them in `bin64\seethroughbubble-cache`.

The settings are in the ReShade overlay (Home key) under the See-Through Bubble tab. F9 turns the bubble on and off.

## Good to know

Set `SEETHROUGH_DEBUG` to 1 at the top of `src/addon.cpp` if you want the status text, the position dots and the extra log lines.

The add-on writes to `bin64\ReShade.log`, so look for lines with `seethroughbubble` when something goes wrong. If a game patch changes the shader layout the add-on depends on, it logs an error and switches the bubble off instead of crashing.

Bump `SEETHROUGH_VERSION` in `src/version.h` when you hand out a new build.

## Credits

Thanks to Carlos Lopez for [RenoDX](https://github.com/clshortfuse/renodx) and crosire for [ReShade](https://github.com/crosire/reshade). This project wouldn't exist without them.

MIT licensed, see [LICENSE](LICENSE).
