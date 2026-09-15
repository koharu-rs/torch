# Torch runtime for Koharu

Prebuilt [LibTorch](https://github.com/pytorch/pytorch) shared libraries and the `koharu-torch` shim for [Koharu](https://github.com/mayocream/koharu).

The release workflow resolves the latest stable PyTorch release and builds CPU, CUDA, HIP, and Metal variants for Windows, Linux, and macOS. It runs on pushes to `main`.

Build jobs use the `ubuntu-latest-xl`, `windows-latest-xl`, and `macos-latest-xl` GitHub-hosted larger runners.

Generated binaries are published through this repository's [releases](https://github.com/koharu-org/torch/releases).

## License

This repository is dual-licensed under the [MIT License](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option.

Third-party code and dependencies retain their respective licenses.
