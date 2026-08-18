# RealMan X5 aligned force capture

This is the GitHub-ready source bundle for the dual RealMan, UGripper, X5
tactile and force-sensor aligned recorder. It replaces the long inline shell
command with a reviewed environment profile while retaining the validated
capture implementation and hardware-free reset/drag tests.

## Scope

Included:

- aligned sampling, freshness checks and transactional episode saving;
- wrist RGB, four X5 tactile streams and bilateral force acquisition;
- foot-pedal start/save/discard controls;
- drag-mode action recording and optional right-arm pre-episode reset;
- stop helper, configuration preflight and unit tests.

Not included:

- captured datasets, videos, logs or model outputs;
- the customized LeRobot checkout and vendor hardware SDKs.

See [the capture stack](docs/ARCHITECTURE.md) for the external LeRobot paths
that must exist under `LEROBOT_ROOT`.

## First-time setup

Use Linux with the compatible LeRobot checkout, the `lerobot51` conda
environment, FFmpeg/NVENC and access to the configured cameras, input device,
robot controllers and UDP ports.

```bash
cp config/lab_5090.env.example config/lab_5090.env
```

Edit only the ignored local file and set:

```bash
LEROBOT_ROOT=/home/your-user/lerobot-git/your-compatible-lerobot
```

Validate without opening hardware:

```bash
scripts/run_capture.sh --check
scripts/verify_bundle.sh
```

The preflight checks the configured device paths, FFmpeg NVENC support and
imports from the companion LeRobot checkout. It does not connect to the arms,
cameras, grippers or force sensors.

## Capture

```bash
scripts/run_capture.sh
```

The runner loads `config/lab_5090.env`, gracefully stops a previous capture,
and starts the aligned recorder. The prepared ignored local profile keeps
datasets in the existing `kd-tacmae/dataset` directory; the tracked example
uses this repository's ignored `dataset/` directory. Select another profile
with:

```bash
scripts/run_capture.sh --config /path/to/profile.env
```

Stop manually with:

```bash
scripts/stop_capture_app.sh
```

The pasted production command did not enable automatic reset, so the committed
profile makes that behavior explicit. To reset the right arm before every
episode, first read [the safety notes](docs/SAFETY.md), clear the workspace and
change this only in the ignored local profile:

```bash
EPISODE_RIGHT_ARM_RESET_ENABLED=true
```

## GitHub publication

This directory is designed to be its own repository. Its upstream repository
is:

```bash
git remote add origin https://github.com/1831571637/capture.git
git push -u origin main
```

The local profile, datasets and generated media are ignored. GitHub Actions
performs shell syntax checks, Python compilation and 11 hardware-free
reset/drag tests; it never starts capture hardware.

The companion LeRobot checkout must also be placed under source control before
deploying this recorder on another machine; it is intentionally not vendored
inside this small capture repository.
