# Alternative models and experimental Vision

QUASAR is the recommended Text and optional TP2 image/video model. This page retains Q4_K_M conversion and the QUASAR Vision startup route. These profiles have their own numerical and memory constraints.

TP2 Vision is opt-in. QUASAR passes the existing TP2/MTP4 prefix integration test at 4096
capacity, including the visual-prefix bridge, changed/appended media and stop/resume. The
older official-NVFP4 profile has a separately unresolved cached/cold greedy difference.
The launch profile below uses cold requests with `--no-prefix-reuse`; users may enable
QUASAR prefix reuse manually. TP4 Vision and DFlash plus Vision are unsupported.

For the alternative Q4_K_M profile, convert the local LM Studio source once (Python 3.11 with NumPy):

```bash
python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --out /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
```

Run Q4_K_M with the unchanged launcher defaults (devices `0,1`, 180K capacity, 4K prefill chunks, INT8 KV,
MTP3 and optimized draft head):

```bash
tools/v100/ninfer-v100x2.sh \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 128 --greedy --no-thinking
```

Set `NINFER_V100X2_ARTIFACT`, `NINFER_V100X2_DEVICES`, `NINFER_V100X2_MAX_CONTEXT`,
`NINFER_V100X2_PREFILL_CHUNK`, `NINFER_V100X2_KV_DTYPE`, or
`NINFER_V100X2_DRAFT_TOKENS` to override the launcher defaults. The launcher does not constrain
host CPU affinity; benchmark CPU use should remain below the operator's 85% ceiling.

### Experimental TP2 Vision

Use the QUASAR v3 artifact; GGUF-derived Q4_K_M cannot enable Vision. The profile below
uses 98K text capacity and 4096 visual tokens, sufficient for the bundled five-second video.
A chart image also passes at 180K capacity with 1024 visual tokens; these are capacity
configurations, not measurements of an equally occupied prompt. `--vision-max-tokens` bounds merged visual tokens, while the complete
text-plus-media prompt must also fit `--max-context`.

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 98304 --kv-capacity 98304 --kv-dtype int8 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --vision --vision-max-tokens 4096 --no-prefix-reuse --default-max-tokens 512 --no-thinking
```

This disables Engine prefix reuse for all requests on this server, including text. It does not
disable the separate immutable-media preprocessing cache. The recommended QUASAR text profile
keeps prefix reuse. See [multimodal requests](serving.md#multimodal-request) and
[known Vision test limitation](../tests/README.md#experimental-tp2-vision-status).
