# Operational safety

- `EPISODE_RIGHT_ARM_RESET_ENABLED=false` is the committed default. Do not
  enable automatic motion until the workspace is clear and an operator is at
  the emergency stop.
- When enabled, the pre-episode reset commands only the right arm. The left arm
  is not sent a reset target. Drag mode starts only after reset succeeds.
- The configured right-arm target is
  `0.251,-0.385,5.442,90.016,0.627,89.769,0.167` degrees. A start pose more than
  45 degrees away on any joint is rejected before motion.
- Run `scripts/run_capture.sh --check` after every configuration change. This
  does not stop a running process and does not open hardware.
- Keep the capture UI bound to `127.0.0.1` unless network access is explicitly
  protected. The UI is not an authentication boundary.
- `scripts/stop_capture_app.sh` sends a graceful stop first, then escalates to
  TERM and KILL for matching capture processes. Do not run it while another
  intended collection job is active on the same host.
- Never place credentials, private SSH keys, datasets, videos or local `.env`
  files in Git.
