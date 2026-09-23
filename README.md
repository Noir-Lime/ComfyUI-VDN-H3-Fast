# ComfyUI VDN H3 Fast

A single **MODEL → MODEL** custom node for using the released VDN-H3 8-step stage in an existing native MiniMax-H3 workflow. It combines a fused CuTe temporal feature operation, existing fused linear-branch helpers, and shared global/anchor attention preparation. The local and nonlocal attention calls use the **released Comfy Kitchen dense INT8 backend**; no ComfyUI or Comfy Kitchen source patch or locally rebuilt Kitchen is needed.

**This optimizes VDN, not native PDD or TaoMate.** It replaces your workflow's other VDN application node; do not stack VDN, PDD, TM, cache or sparse-attention patches on the same MODEL. Leave your reference, audio, VAE, sampler and save nodes in place. Reconnect just the MODEL path:

```text
MiniMax-H3 UNET Loader ──► VDN H3 Fast ──► Basic Guider ──► SamplerCustomAdvanced
                              checkpoint: stage-dmd-step-250-int8_convrot_comfyui
```

Use **Euler + 8-step Simple schedule** with the released DMD stage. The node applies both stage adapters at their trained strengths (1.0), checkpoint-defined hybrid architecture, grouped windows, and the verified optimized paths. The VDN stage changes model behavior and may change generation quality; it is not a drop-in speed boost for existing PDD/TM output.

## Install

1. In ComfyUI Manager, install from Git URL `https://github.com/Noir-Lime/ComfyUI-VDN-H3-Fast`, or clone it into `ComfyUI/custom_nodes/ComfyUI-VDN-H3-Fast`.
2. Install `requirements.txt` with **ComfyUI's own Python**, then restart ComfyUI. It installs NVIDIA CuTe DSL; ComfyUI supplies PyTorch, safetensors, and Comfy Kitchen. Tested with PyTorch 2.13.0+cu130 and comfy-kitchen 0.2.35 on RTX 5090. NVIDIA CUDA/BF16 is the accelerated target; on unsupported devices the native temporal operation and attention fallbacks run instead. The first execution may compile the two small existing fused helpers and the CuTe kernel.
3. Put a released VDN stage directory under `ComfyUI/models/vdn/`; preserve `model_spec.json`, `linear_branch/`, and `adapters/`. The [pre-quantized INT8 ConvRot DMD stage](https://huggingface.co/drbaph/vdn-minimax-h3-int8-convrot-comfyui) is the tested option. Model weights are **not** distributed here. The node uses the INT8 branch if present, otherwise the original BF16 branch.
4. **Pruned/curve MiniMax-H3 bases:** place `h3_silu_temb_grid.safetensors` under `models/vdn/`, or install [ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo), which includes that grid. The grid is used to re-inject all trained AdaLN adapter deltas; this node errors if it is missing rather than silently dropping them. Unpruned H3 bases do not require it. The grid is not a model weight and comes from the Turbo project's repository.
5. Add **VDN H3 Fast** to your existing H3 graph. Connect the base H3 MODEL to its input and its output to the existing guider/sampler. Don't connect PDD/TM LoRA loaders before it.

The upstream VDN checkpoint and MiniMax-H3 weights have their **own licenses**, separate from this Apache-2.0 code. Check the model terms for your location and use; nothing downloads weights automatically.

## Evidence and limitations

At 768×448, 362 frames, native 8-step DMD stage, RTX 5090 capped at 400 W, fixed seed and references: the tested isolated implementation reduced sampler time from **88.3 s to 82.6 s** compared with its CuTe-only VDN control. The CuTe change reduced total time by 5.6% versus the earlier VDN baseline; the fused helpers and nonlocal batching gave another ~6.4% sampler gain. These are single-run measurements, not cross-hardware guarantees. The fused helpers change rounding; decoded output was visually comparable in sampled frames and audio correlation was 0.9957, but listening was not verified. Batching added no decoded-media difference beyond the fused helpers in that comparison. Native PDD/TM + Safe Sol still sampled faster (~58 s) on this same scene.

The standalone node was tested in the resident ComfyUI installation on the same 15-second scene. **All 362 decoded video frames and decoded audio matched the isolated optimized VDN result exactly** (SSIM 1.0; decoded stream hashes equal), and its eight-step sampler progressed at approximately 10.3 s/step, consistent with the isolated 82.6-second sampler. Its first cold server execution took 113.3 s including model/text-encoder loading and first compilation; do not compare that directly with the warmed 91.6-second isolated total. This is one machine and one scene, not a clean-install or cross-hardware guarantee. The whole model is not compiled; only select helper operations and the temporal kernel are. Low-VRAM success is model/shape dependent.

The native stage files, personal reference images/audio and generated media are deliberately excluded. [NOTICE](NOTICE) preserves source attribution to OpenVDN, Saganaki22, xmarre and the original MiniMax-H3-Turbo adapter integration.
