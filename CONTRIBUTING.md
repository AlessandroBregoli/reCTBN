# How to Contribute

Thank you for considering contributing. All contributions are welcome and
appreciated!

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

## Bug Reports

> **Note:** Be sure to have censored any sensitive data before posting,
> especially in the logs!

**Github issues** is used for tracking bugs and proposing new features. Please
consider the following when opening an issue:

+ Avoid opening duplicate issues by taking a look at the current open issues.
+ Follow [this template](.github/ISSUE_TEMPLATE/bug_report.md).
+ Provide details on the library version, operating system and Rust version you
  are running.
+ Provide the **expected** and the **actual** behavior.
+ Provide steps to reproduce the issue
+ Include complete log tracebacks and error messages.
+ An optional description to give more context

## Feature Proposal

**Github issues** is used for tracking bugs and proposing new features. Please
consider the following when opening an issue:

+ Avoid opening duplicate issues by taking a look at the current open and closed
  ones.
+ Follow [this template](.github/ISSUE_TEMPLATE/feature_request.md).
+ The feature must be concrete enough to visualize, broken into parts, easy to
  manage and the task not too heavy.
+ Provide a short description from the point of view the actor(s) who will
  benefit from it, specifying what feature you want and what the actors should
  achieve with it. In other words you have to **justify** why the feature can be
  beneficial for the project.
+ Provide a list of acceptance criteria.
+ If you have clear ideas on how implement this feature write a checklist (not
  too much precise but also not too much vague) on changes you have to bring to
  the project.
+ To give more context you can optionally add a (not too long) description with
  images attached.

## Pull Requests

All pull requests are welcome, but please consider the following:

+ **You cannot merge into master**, merge in **dev** instead.
+ Follow [this template](.github/pull_request_template.md).
+ Please open an issue first if a relevant one is not already open.
+ Include tests.
+ Include documentation for new features.
+ If your patch is supposed to fix a bug, please open an issue first.
+ Avoid introducing new dependencies.
+ Clarity is preferred over brevity.
+ Please follow `cargo check`.
+ Before the merge the code will be reviewed and should pass an acceptance
  testing from other contributors and/or project leaders.
+ Provide a changelog (not too much detailed) about your changes

> **Note:**
>
> + The merge requests to _master_ **_must only come from_** the _dev_ branch.
> + In case an **hotfix** is necessary you can only _branch from master_, apply
>   the fix (maybe cherry-picking from other branches) and then merge back to
>   master.
> + For contributors who have write permissions to master
>   [here the template](.github/PULL_REQUEST_TEMPLATE/merge_master.md) for the
>   merge request.
