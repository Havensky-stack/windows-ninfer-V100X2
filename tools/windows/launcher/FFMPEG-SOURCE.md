# Corresponding FFmpeg source for the Windows preview

`ffmpeg-ffmpeg-n8.1.2.tar.gz` is the complete source for FFmpeg n8.1.2.
`vcpkg-port/` contains the exact 8.1.2#3 port and all applied patches/build scripts.
The port's SHA512 verifies this source archive. FFmpeg is compiled as replaceable DLLs,
with LGPL configuration; GPL and nonfree features are disabled.

To rebuild, use x64 Windows, MSVC 14.44, Ninja and vcpkg at baseline
`4bca8fd8654e5ba76f92661db7bfe954768ad8ef`. Set `VCPKG_DEFAULT_BINARY_CACHE` to a separate
directory if you want to force a source build instead of reusing an installed archive.
The provided manifest and `x64-windows-ninfer.cmake` triplet select dynamic Release libraries
and dynamic CRT. Run from this directory with the MSVC x64 developer environment loaded:

```powershell
vcpkg install --triplet x64-windows-ninfer --overlay-triplets=.
```

The fixed manifest selects FFmpeg avcodec/avformat/swscale/zlib, without default features.
For a modified version, place this `vcpkg-port` directory under `ports/ffmpeg`, make your
changes there, and add `--overlay-ports=ports` to the command. vcpkg downloads its build helpers
(including NASM and ffmpeg-bin2c) automatically. The output DLLs are under
`vcpkg_installed/x64-windows-ninfer/bin`; compatible replacements may be copied over the
launcher package's `bin/avcodec-62.dll`, `avformat-62.dll`, `avutil-60.dll`, and `swscale-9.dll`.

The original build used Visual Studio 2026's MSVC 14.44.35207 compiler and Windows SDK
10.0.26100.0. The source/port are also available at:

- https://github.com/FFmpeg/FFmpeg/tree/n8.1.2
- https://github.com/microsoft/vcpkg/tree/4bca8fd8654e5ba76f92661db7bfe954768ad8ef/ports/ffmpeg

Full engine source/build instructions: https://github.com/Havensky-stack/windows-ninfer-V100X2
