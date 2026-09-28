"""Local MLX speech curation: prepare, score, calibrate, select, export."""

import argparse
import json
from pathlib import Path

from .pipeline import prepare, score, select

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
WHISPER_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
VAD_MODEL = "mlx-community/silero-vad"
VAD_REVISION = "7bc17f22d3c0451bd3a6cd71e759b009271ff49a"


def default_stages(args):
    from .alignment import QwenAligner, TurkishAligner
    from .backends import WhisperASR
    from .quality import DNSMOS
    from .speakers import ECAPASpeaker, NemotronDiarizer
    from .stages import Stage

    factories = {
        "asr": lambda: WhisperASR(args.model, args.revision, local_files_only=args.offline),
        "quality": lambda: DNSMOS(args.dnsmos_model, local_files_only=args.offline),
        "diarization": lambda: NemotronDiarizer(local_files_only=args.offline),
        "speaker": lambda: ECAPASpeaker(local_files_only=args.offline),
        "alignment_qwen": lambda: QwenAligner(local_files_only=args.offline),
        "alignment_tr": lambda: TurkishAligner(local_files_only=args.offline),
    }
    names = args.stages.split(",")
    if set(names) - factories.keys() or len(set(names)) != len(names) or names[0] != "asr":
        raise ValueError("--stages must start with asr and contain unique supported stages")
    return [
        Stage(
            name,
            factories[name],
            ("tr",)
            if name == "alignment_tr"
            else ("en", "zh", "ja")
            if name == "alignment_qwen"
            else ("en", "zh", "ja", "tr"),
        )
        for name in names
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--model", default=WHISPER_MODEL if name == "score" else VAD_MODEL)
        command.add_argument(
            "--revision", default=WHISPER_REVISION if name == "score" else VAD_REVISION
        )
        command.add_argument("--offline", action="store_true")
        if name == "score":
            command.add_argument(
                "--dnsmos-model", type=Path, help="pinned local DNSMOS ONNX conversion source"
            )
            command.add_argument("--retry-errors", action="store_true")
            command.add_argument(
                "--stages", default="asr,quality,diarization,speaker,alignment_qwen,alignment_tr"
            )
    selection = commands.add_parser("select")
    selection.add_argument("database", type=Path)
    selection.add_argument("--policy", type=Path, required=True)
    selection.add_argument("--profile", choices=("tts", "asr"), default="asr")
    selection.add_argument("--out", type=Path, required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("database", type=Path)
    calibration.add_argument("--out", type=Path, required=True)
    calibration.add_argument("--per-group", type=int, default=20)
    calibration.add_argument("--seed", default="0")
    calibration.add_argument("--labels", type=Path)
    calibration.add_argument("--policy", type=Path)
    calibration.add_argument("--profile", choices=("tts", "asr"))
    export = commands.add_parser("export")
    export.add_argument("selection", type=Path)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument(
        "--sample-rate", required=True, help="preserve or explicit training sample rate"
    )
    export.add_argument("--seed", default="0")
    export.add_argument("--train", type=float, default=0.9)
    export.add_argument("--validation", type=float, default=0.05)
    args = parser.parse_args()
    try:
        if args.command == "select":
            result = select(args.database, args.policy, args.out, args.profile)
        elif args.command == "calibrate":
            from .curation import calibrate

            result = calibrate(
                args.database,
                args.out,
                per_group=args.per_group,
                seed=args.seed,
                labels=args.labels,
                policy_path=args.policy,
                profile=args.profile,
            )
        elif args.command == "export":
            from .curation import export

            result = export(
                args.selection,
                args.out,
                sample_rate=args.sample_rate,
                seed=args.seed,
                train=args.train,
                validation=args.validation,
            )
        elif args.command == "prepare":
            from .backends import SileroVAD
            from .speakers import NemotronDiarizer

            result = {
                "segments": prepare(
                    args.manifest,
                    args.out,
                    lambda: SileroVAD(args.model, args.revision, local_files_only=args.offline),
                    diarizer_factory=lambda: NemotronDiarizer(local_files_only=args.offline),
                )
            }
        else:
            result = score(
                args.manifest, args.out, stages=default_stages(args), retry_errors=args.retry_errors
            )
        print(json.dumps(result, ensure_ascii=False))
        if result.get("errors", 0) or result.get("error", 0):
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
