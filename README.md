# Torch runtime for Koharu

Prebuilt [LibTorch](https://github.com/pytorch/pytorch) shared libraries and the `koharu-torch` shim for [Koharu](https://github.com/mayocream/koharu).

The shim gives every ATen operator a C function, which `shim/generate.py` writes from the operators of the PyTorch version it builds for, and `shim/koharu_torch.h` the rest of libtorch's C++ API that libtorch's own C functions lack. Each shim release's `include.tar.gz` holds the shim's headers and the libtorch C headers they include.

libtorch and the shim are released apart, each when a tag following [semantic versioning](https://semver.org) is pushed. A tag such as `2.15.0+libtorch.1` builds PyTorch 2.15.0, its last number counting builds of that version, as CPU, CUDA, HIP, and Metal variants for Windows, Linux, and macOS. Each build publishes its libraries and a `-dev` archive of the headers and Windows import libraries the shim builds against. A tag such as `2.15.0+koharu.0.84.0.alpha.1` builds the shim for the Koharu version it names against the newest libtorch release of that PyTorch version.

libtorch's build jobs use the `ubuntu-latest-xl`, `windows-latest-xl`, and `macos-latest-xl` GitHub-hosted larger runners; the shim builds in minutes on standard ones.

Generated binaries are published through this repository's [releases](https://github.com/koharu-org/torch/releases).

## License

This repository is dual-licensed under the [MIT License](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option.

Third-party code and dependencies retain their respective licenses.
