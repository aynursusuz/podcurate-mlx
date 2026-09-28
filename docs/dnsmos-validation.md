# DNSMOS P.835 native MLX validation — 2026-09-28

The production `quality.DNSMOS` adapter executes the pinned nonpersonalized P.835
network with MLX Metal in float32. It does not import ONNX Runtime and has no CPU
neural inference fallback. `ONNXDNSMOS` is an explicit, separate validation adapter.

## Model and attribution

- Model: Microsoft DNSMOS P.835, nonpersonalized `sig_bak_ovr.onnx`, 1,157,965 bytes.
- [Official model snapshot](https://raw.githubusercontent.com/microsoft/DNS-Challenge/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/DNSMOS/sig_bak_ovr.onnx)
  at commit `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`.
- Source SHA-256: `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd`.
- Converted tensor archive SHA-256 observed here:
  `16c1f1600fcf92cbc036678dda7a4d2344d20f2406db6c6941f72e8ee6f4e2b7`.
- [Reference scoring code](https://github.com/microsoft/DNS-Challenge/blob/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/dnsmos_local.py).
- [DNSMOS P.835 paper](https://arxiv.org/abs/2110.01763), Reddy, Gopal and Cutler, ICASSP 2022.
- The pinned upstream repository's [LICENSE is CC BY 4.0](https://github.com/microsoft/DNS-Challenge/blob/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/LICENSE).
  Microsoft is the source of the model and scoring method. This project changes the
  execution backend and tensor layouts; it does not retrain or quantize the weights.
  Model weights are downloaded separately and are not relicensed under this project's MIT license.

## Exact port scope

The implementation handles this graph and checksum only. Local model overrides with a
changed checksum are rejected. It preserves the learned real/imaginary projection
kernels, magnitude/square/log operations, seven convolutions, three valid max pools,
global max pooling and three dense layers. ONNX NCHW activations become MLX NHWC;
ONNX OIHW convolution weights become MLX OHWI. Intermediate comparisons restore the
same layout before measuring differences. No STFT approximation or replacement weights
are used.

Input is unnormalized 16 kHz mono float32. Reference 144,160-sample windows, 16,000-sample
hops, doubling of short input and the original hop-count expression are preserved.
The nonpersonalized SIG/BAK/OVRL calibration polynomials are applied per window and then
averaged. Output accumulation uses float64; the network uses float32. The P.808 model,
personalized variant and resampling are outside this adapter.

`DNSMOS()` downloads the commit-pinned model, verifies SHA-256 and converts weights on
its first call to the constructor. `DNSMOS('/path/sig_bak_ovr.onnx')` uses a local copy.
Conversion requires `onnx==1.23.0`; subsequent cached inference needs MLX and NumPy,
and is tested with both `onnx` and `onnxruntime` imports actively blocked. Reference
comparison additionally needs `onnxruntime`. Cache location is
`$PODCURATE_CACHE_DIR/dnsmos`, or `$XDG_CACHE_HOME/podcurate-mlx/dnsmos`, with
`~/.cache` used when XDG_CACHE_HOME is absent. Download/conversion writes use atomic
replacement. Model identity records source, converted and graph-code hashes, graph revision,
MLX version, backend and dtype. `close()` releases weights and the MLX cache.
`local_files_only=True` forbids model download; a missing cached model fails before
any network request. An explicit local model path remains usable in offline mode.

## Executed checks

Machine: Apple M4 Pro, 24 GiB, macOS 26.6.2 arm64, Python 3.12.13.
Packages: MLX 0.32.2, NumPy 2.5.3, ONNX 1.23.0, ONNX Runtime 1.30.0.
Reference: CPUExecutionProvider, one intra-op thread. GPU checks were serialized.

All 18 observed intermediate tensors were compared for silence, seeded Gaussian noise
and the first scoring window of each of 16 real FLEURS recordings. Calibrated clip-level
SIG/BAK/OVRL scores were compared across all reference windows: 60 windows in the speech
cohort. Recordings total 188.76 seconds. The source recordings are read speech, not
podcasts or synthetic speech. The `kind` value used by a pipeline fixture does not change
their real provenance.

| Input | Recordings | Scored windows | Max absolute raw output difference, observed first windows | Max absolute calibrated score difference | Max absolute intermediate tensor difference |
|---|---:|---:|---:|---:|---:|
| en | 4 | 13 | 4.7683716e-07 | 6.1383441e-07 | 6.6757202e-06 |
| zh | 4 | 18 | 9.5367432e-07 | 4.1564421e-07 | 4.7683716e-06 |
| ja | 4 | 12 | 9.5367432e-07 | 2.9509143e-07 | 3.5762787e-06 |
| tr | 4 | 17 | 9.5367432e-07 | 2.9634464e-07 | 4.2915344e-06 |
| silence | 1 | 1 | 7.1525574e-07 | 5.3138847e-07 | 3.8146973e-06 |
| noise | 1 | 1 | 5.9604645e-07 | 6.2361286e-07 | 1.3113022e-06 |

Noise input: NumPy `default_rng(314159).normal(0, 0.05, 144160).astype(float32)`;
silence input: 144,160 zero float32 samples. Full calibrated control outputs:

| Input | MLX SIG | MLX BAK | MLX OVRL | ONNX SIG | ONNX BAK | ONNX OVRL |
|---|---:|---:|---:|---:|---:|---:|
| silence | 2.5135647012 | 3.4724233332 | 1.8398629311 | 2.5135648931 | 3.4724238646 | 1.8398628272 |
| noise | 1.1836394099 | 1.1005963340 | 1.1152592668 | 1.1836400335 | 1.1005966540 | 1.1152593833 |

Test tolerances are numerical port tolerances, not quality-selection thresholds:
intermediate `atol=2e-5, rtol=2e-4`; calibrated output absolute tolerance `2e-5`.
Observed maximum intermediate error was `6.67572021484375e-6`; maximum calibrated error
across speech and controls was `6.236128564651722e-7`. The native path's observed peak MLX
allocation for clip scoring was 153,149,076 bytes, excluding tensor tracing. This is an
MLX allocator measurement, not process RSS or whole-pipeline memory. No corpus throughput
claim is inferred from these small fixtures.

The model-enabled run of `tests/test_quality.py` completed **13 tests** with the real
model and four-language fixtures. A subsequently added offline-resolution regression
also passed; the current weights-free suite is **11 passed, 3 skipped**. Ruff passed.
Tests include real layer/output parity, reference window boundaries, short input repeat,
empty/nonfinite/stereo rejection, close behavior, bad source/download checksums, and
cached native inference while ONNX/ONNX Runtime imports are forbidden. The default
constructor was also run against an empty cache: pinned download, verification,
automatic conversion and real Metal inference all passed.

## Reproduction

Install the project's MLX quality, converter and validation dependencies. Provide the
pinned ONNX model locally and the fixture WAVs described below; the test suite does not
download models or audio. From the repository root:

```bash
PODCURATE_DNSMOS_TEST_MODEL=/absolute/path/sig_bak_ovr.onnx \
PODCURATE_DNSMOS_TEST_FIXTURES=/absolute/path/fleurs-fixtures \
python -m pytest tests/test_quality.py -q
```

The fixture directory must contain `manifest.jsonl` with `audio`, `language`, `id` and
`sha256` per row. Test decoding uses FFmpeg at 16 kHz mono. FLEURS snapshot:
[`google/fleurs` at `70bb2e84b976b7e960aa89f1c648e09c59f894dd`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd).
Original WAV names below identify members of `data/<locale>/audio/test.tar.gz`.
Locale mapping: en→en_us, zh→cmn_hans_cn, ja→ja_jp, tr→tr_tr. FLEURS attribution and
CC BY 4.0 provenance remain attached to the fixture manifest.

| Local WAV | Original WAV | SHA-256 |
|---|---|---|
| en-0.wav | 1003119935936341070.wav | `33aca50159ec2e3cbcc894eb26faa916d6badb1b49fb75994973561540a7f012` |
| en-1.wav | 10052240106321793346.wav | `7603cae3294309ef70d234180cefccc3b13865ee525262228a636069b9b62841` |
| en-2.wav | 10167324587744183095.wav | `260bdd1a15e1d35bcc5d91667516ca3bc659285e10f94872182715bcd7880398` |
| en-3.wav | 10197164397713068203.wav | `b174be7730f0aab6a26762130b66f2125392b50550645727bd057d8479fb6219` |
| zh-0.wav | 10026684690566417990.wav | `a72c0b59fdba80850552af7991de07f9f4940846d2d70f0890eeb153c40f164b` |
| zh-1.wav | 10040380210557600780.wav | `b39d2c84aefada280b73a9ab439726d370d45e5a75bd05f0a90153288ea4561b` |
| zh-2.wav | 10048525650290665384.wav | `aeb95366f4907daf8797d66797c87f6e002f10704e78e87a0c4b4521756a1616` |
| zh-3.wav | 10053956375630517392.wav | `d1274fd5e5ce0029473faa0df42bd714094428b853029d6f6c64ffd3f36a8afe` |
| ja-0.wav | 10020345318418093976.wav | `bd1b9533cfa176b8c2a71ecb4821a02bc86fa368d096373efe8d8dceafb35634` |
| ja-1.wav | 10086073819084529371.wav | `fa4d3ce475c9b2fd127d3c9a6ea445792e21f04c5f8514452b0ad97b0ccbc2f8` |
| ja-2.wav | 10112379500600537933.wav | `ade60c57813d704095310734906a9b6918b2dbbdcd0c9405c3f7d25580570d48` |
| ja-3.wav | 10174984673310355687.wav | `7059c2e63434b2ffb6d33782ba99bf31b2c3f725feb988d50c514adda8efdb64` |
| tr-0.wav | 10000377651956138413.wav | `e2efd9f3ce2fcc9e191126dcded8e674233a9ce83a9c2132541700e8f1bd57c3` |
| tr-1.wav | 10002518854052673247.wav | `8d49d9d174794b25da72852685120ff5dd1d7319ca629256fafd1e25c635fc9a` |
| tr-2.wav | 10067972835475076305.wav | `0d35468a6dec8161711e470404f64471ceed7d7197adcd2e7638a4fc3182620d` |
| tr-3.wav | 10073710776711594036.wav | `71c90819a4b8f5c5ade78626e921d7efe27fa087cf1e17581d1242a6f26ca5de` |

## What this establishes

These checks establish numerical agreement of this graph port on the stated inputs.
They do not calibrate DNSMOS against human judgments in en/zh/ja/tr, establish quality
cutoffs, measure synthetic artifacts/identity/content fidelity, or validate downstream
TTS/ASR training improvements. The silence score itself illustrates why a finite MOS
estimate must not be treated as proof of usable speech. Calibration on the user's
labeled audio remains a separate step.
