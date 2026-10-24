#!/usr/bin/env python3
"""Bundle the explicit SM70 release build and its non-driver Linux runtime dependencies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile


SYSTEM_LIBS = {
    'libc.so.6', 'libm.so.6', 'libdl.so.2', 'libpthread.so.0', 'librt.so.1',
    'libresolv.so.2', 'libutil.so.1', 'ld-linux-x86-64.so.2', 'libcuda.so.1',
}


def external(name: str) -> bool:
    return name in SYSTEM_LIBS or name.startswith('libnvidia-')


def dependencies(binary: Path, env: dict) -> dict[str, Path]:
    result = subprocess.run(['ldd', str(binary)], env=env, capture_output=True, text=True, check=True)
    if 'not found' in result.stdout:
        raise RuntimeError(f'Unresolved dependency for {binary}:\n{result.stdout}')
    return {match[1]: Path(match[2]) for match in re.finditer(r'^\s*(\S+) => (/\S+)', result.stdout, re.M)
            if not external(match[1])}


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--dependency-prefix', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda-12.8'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if platform.machine() != 'x86_64' or platform.system() != 'Linux':
        parser.error('this packager targets Linux x86_64')
    if not re.fullmatch(r'\d+\.\d+\.\d+', args.version):
        parser.error('--version must be a numeric release version, such as 0.2.0')
    build = args.build.resolve()
    cache = (build / 'CMakeCache.txt').read_text()
    if not re.search(r'^CMAKE_CUDA_ARCHITECTURES:[^=]+=70$', cache, re.M):
        parser.error('only an explicit SM70 build can be bundled')
    name = f'ninfer-{args.version}-linux-x86_64-sm70-cuda12.8'
    destination = args.output.resolve() / name
    destination.mkdir(parents=True, exist_ok=False)
    for directory in ('bin', 'libexec', 'lib', 'licenses', 'tools/v100'):
        (destination / directory).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = f'{args.dependency_prefix.resolve()}/lib:{args.cuda.resolve()}/lib64'
    libraries: dict[str, Path] = {}
    executables = {}
    for binary, relative in [('ninfer', 'apps/ninfer'), ('ninfer-serve', 'apps/ninfer-serve'),
                             ('ninfer_bench', 'bench/ninfer_bench')]:
        source = build / relative
        if not source.is_file():
            raise RuntimeError(f'Missing release executable: {source}')
        executables[binary] = relative
        libraries.update(dependencies(source, env))
        target = destination / 'libexec' / binary
        shutil.copy2(source, target)
        subprocess.run(['strip', '--strip-unneeded', str(target)], check=True)
        write_executable(destination / 'bin' / binary, '#!/bin/sh\nset -eu\n'
            'root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\n'
            'export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
            f'exec "$root/libexec/{binary}" "$@"\n')
    for soname, source in sorted(libraries.items()):
        shutil.copy2(source.resolve(), destination / 'lib' / soname)
    # Include the launchers and the two shipped user configurations, without model weights.
    for file in ('ninfer_tui.py', 'ninfer_gui.py', 'gui_cache.py', 'launcher_sampling.py', 'launcher_model.py',
                 'launcher_configs.py', 'launcher_resources.py'):
        shutil.copy2(repo / 'tools/v100' / file, destination / 'tools/v100' / file)
    shutil.copytree(repo / 'tools/v100/configs', destination / 'tools/v100/configs')
    default_file = destination / 'tools/v100/configs/defaults.json'
    defaults = json.loads(default_file.read_text())
    for values in defaults['profiles'].values():
        values['binary'] = 'bin/ninfer-serve'
    default_file.write_text(json.dumps(defaults, ensure_ascii=False, indent=2) + '\n')
    for launcher, file in [('ninfer-tui', 'ninfer_tui.py'), ('ninfer-gui', 'ninfer_gui.py')]:
        write_executable(destination / 'bin' / launcher, '#!/bin/sh\nset -eu\n'
            'root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\n'
            f'exec "${{NINFER_PYTHON:-python3}}" "$root/tools/v100/{file}" '
            '--binary "$root/bin/ninfer-serve" "$@"\n')
    for file in ('LICENSE', 'NOTICE'):
        shutil.copy2(repo / file, destination / file)
    (destination / 'docker').mkdir()
    for file in ('entrypoint.py', 'Dockerfile.release'):
        shutil.copy2(repo / 'docker' / file, destination / 'docker' / file)
    shutil.copytree(repo / 'third_party/licenses', destination / 'licenses/project-dependencies')
    # Distribution copyright files include the license and required attribution for system DSOs.
    missing_licenses = []
    for source in sorted(set(libraries.values())):
        if source.name.startswith('libcudart') or args.dependency_prefix.resolve() in source.resolve().parents:
            continue
        result = subprocess.run(['dpkg-query', '-S', str(source.resolve())], capture_output=True, text=True)
        if result.returncode:
            result = subprocess.run(['dpkg-query', '-S', str(source)], capture_output=True, text=True)
        packages = {line.split(': ', 1)[0].split(':')[0] for line in result.stdout.splitlines() if ': ' in line}
        found = False
        for package in packages:
            license_file = Path('/usr/share/doc') / package / 'copyright'
            if license_file.is_file():
                shutil.copy2(license_file, destination / 'licenses' / f'{package}.copyright')
                found = True
        if not found:
            missing_licenses.append(str(source))
    dependency_sources = args.dependency_prefix.resolve().parent / 'sources'
    # Ship the unmodified LGPL runtime source beside the binary asset at the same release.
    ffmpeg_source = dependency_sources / 'ffmpeg-6.1.2.tar.xz'
    if not ffmpeg_source.is_file():
        raise RuntimeError(f'Missing bundled FFmpeg source archive: {ffmpeg_source}')
    source_archive = destination.parent / ffmpeg_source.name
    shutil.copy2(ffmpeg_source, source_archive)
    for directory, license_name in [('ffmpeg-6.1.2', 'COPYING.LGPLv2.1'), ('curl-8.10.1', 'COPYING')]:
        source = dependency_sources / directory / license_name
        if source.is_file():
            shutil.copy2(source, destination / 'licenses' / f'{directory}.txt')
        else:
            raise RuntimeError(f'Missing bundled dependency license: {source}')
    cuda_license = args.cuda / 'EULA.txt'
    if not cuda_license.is_file():
        raise RuntimeError(f'Missing CUDA runtime redistribution terms: {cuda_license}')
    shutil.copy2(cuda_license, destination / 'licenses/CUDA-EULA.txt')
    if missing_licenses:
        raise RuntimeError(f'Missing distribution copyright files: {missing_licenses}')
    packaged_env = dict(os.environ, LD_LIBRARY_PATH=str(destination / 'lib'))
    glibc_versions = {(2, 17)}
    for binary in list((destination / 'libexec').iterdir()) + list((destination / 'lib').iterdir()):
        for soname, resolved in dependencies(binary, packaged_env).items():
            if resolved.parent != destination / 'lib':
                raise RuntimeError(f'Unbundled runtime dependency: {soname} -> {resolved}')
        versions = subprocess.run(['readelf', '--version-info', str(binary)], capture_output=True,
                                  text=True, check=True).stdout
        glibc_versions.update(tuple(map(int, v.split('.'))) for v in re.findall(r'\bGLIBC_(\d+\.\d+)\b', versions))
    manifest = {
        'version': args.version, 'platform': 'linux-x86_64', 'cuda': '12.8', 'sm': 70,
        'minimum_glibc': '.'.join(map(str, max(glibc_versions))),
        'tested_distribution': 'Ubuntu 22.04', 'recommended_driver': '570 or newer',
        'weights_included': False, 'executables': executables,
        'bundled_libraries': sorted(libraries),
        'external_libraries': sorted(SYSTEM_LIBS),
        'cpu_blas': 'OpenBLAS 0.3.30, static, DYNAMIC_ARCH=1, USE_THREAD=0',
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(repo / 'docs/prebuilt.md', destination / 'README.md')
    # Help must work with only the package directory on the dynamic library path.
    for executable in ('ninfer', 'ninfer-serve', 'ninfer_bench'):
        result = subprocess.run([str(destination / 'bin' / executable), '--help'],
                                env=packaged_env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'{executable} failed package smoke: {result.stderr}')
    archive = Path(str(destination) + '.tar.gz')
    with tarfile.open(archive, 'w:gz', compresslevel=6) as bundle:
        bundle.add(destination, arcname=destination.name)
    print(json.dumps({'directory': str(destination), 'archive': str(archive),
                      'runtime_source_archive': str(source_archive), 'manifest': manifest}, indent=2))


if __name__ == '__main__':
    main()
