---
title: Reachy Mini Motion Generator
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
tags:
  - reachy_mini
  - text-to-motion
  - robotics
short_description: Type a prompt, watch Reachy Mini perform it in 3D
---

# Reachy Mini Motion Generator

Type what Reachy Mini should express, for example *"sneezing. You build up and then sneeze loudly."*, and watch the motion
play in 3D, rendered entirely in your browser.

- **Endpoint.** Point the page at a running `inference.server` (`POST /generate-dense`, prompt in, trajectories out). You can
  also pass it in the URL as `?endpoint=https://…`. The page is HTTPS, so the endpoint must be HTTPS as well.
- **No endpoint?** The examples are real outputs of the fine-tuned Qwen3.8-27B planner, pre-generated.
- **Any move.** Drop a Reachy Mini move JSON (the SDK recorded-move format, e.g. from the emotions library) onto the page.

How it renders: each frame's head pose, antennas and body yaw are turned into the six Stewart-platform motor angles by
a JavaScript port of the SDK's analytical IK (`src/StewartIK.js`, verified against the Rust implementation to 1e-12 rad).
The passive joints then follow from the 3D model's kinematics.

Credits: the 3D model, meshes, scene and passive-joint kinematics come from
[8bitkick/reachy_mini_3d_web_viz](https://huggingface.co/spaces/8bitkick/reachy_mini_3d_web_viz) (Apache-2.0,
`LICENSE-reachy_mini_3d_web_viz.txt`). The IK is ported from
[pollen-robotics/reachy_mini_rust_kinematics](https://github.com/pollen-robotics/reachy_mini_rust_kinematics).
