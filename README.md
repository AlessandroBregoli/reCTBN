<div align="center">

# reCTBN

</div>

## Library

> **Note:** At the moment it's in pre-alpha state.

A Continuous Time Bayesian Networks Library written in Rust. 🦀

## Develop

**Prerequisites:**

+ `rust`

_Prepare_ the development environment:

```sh
cargo build
```

That's all! ☕️

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) to know how to report bugs, propose
features, merge requests or other forms of contribution! 😎🚀

## Testing & Linting

To launch **tests**:

```sh
cargo test
```

To **lint** with `cargo check`:

```sh
cargo check --all-targets
```

Or with `clippy`:

```sh
cargo clippy --all-targets -- -A clippy::all -W clippy::correctness
```

To check the **formatting**:

> **NOTE:** remove `--check` to apply the changes to the file(s).

```sh
cargo fmt --all -- --check
```

## Documentation

To generate the **documentation**:

```sh
cargo rustdoc --package reCTBN --open -- --default-theme=ayu
```

## License

This software is distributed under the terms of both the Apache License
(Version 2.0) and the MIT license.

See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for
details.
