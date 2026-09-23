# Face

A procedural human-head construction experiment for [CUDA WebShader](https://github.com/SamG-Coder/cuda-webshader).

The goal is not to load a head mesh. The browser starts from face-construction parameters, CUDA generates a fixed-topology head directly into WebGPU storage buffers, and a raw WebGPU renderer draws those same buffers without CPU readback.

## What is in the first milestone

- Front/profile construction drawings driven by the same semantic parameters as the 3D head.
- CUDA-authored head surface with brow, eye sockets, cheeks, nose, muzzle, lips, jaw and chin.
- CUDA-generated vertex positions and normals.
- Fixed topology suitable for later identity/expression layers.
- Raw WebGPU clay renderer.
- Smooth, planar/faceted and wireframe inspection modes.
- Orbit camera and live anatomy controls.
- Deterministic random-face generation.
- No imported model, no Three.js, no render framework.

## Run

Requirements: Node 20+, Git, and a WebGPU-capable browser.

```bash
npm start
```

Open http://localhost:5173.

The start script syncs the compiler/runtime source from `SamG-Coder/cuda-webshader` into a generated `vendor/` directory. Only the WebShader compiler/runtime is copied; this project does not use Three.js.

## Build

```bash
npm run build
```

The static site is written to `dist/`.

## Pipeline

```text
semantic face controls
        |
        +--> front/profile construction canvas
        |
        +--> kernels/face.cu
                |
                v
        CUDA WebShader compiler
                |
                v
        WebGPU compute
          positions + normals
                |
                v
        raw WebGPU render pipeline
```

The compute and render pipelines share the same `GPUDevice` and the renderer reads the CUDA-written storage buffers directly.

## Next anatomy milestones

The current head establishes the construction system first. The next steps are separate procedural patches for eyelids around eyeballs, nasal wings/nostrils, lips with a real mouth opening, ears, then identity/expression/asymmetry layers and skin shading.
