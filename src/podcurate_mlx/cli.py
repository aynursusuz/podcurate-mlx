"""Three commands: prepare podcast spans, score, select using an explicit policy."""

import argparse
import json
from pathlib import Path

from .pipeline import prepare, score, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("--out", type=Path, required=True)
        default = ("mlx-community/whisper-large-v3-turbo" if name == "score"
                   else "mlx-community/silero-vad")
        command.add_argument("--model", default=default)
        command.add_argument("--revision", default="main",
                             help="use a model commit SHA for exact repeatability")
        command.add_argument("--offline", action="store_true")
        if name == "score":
            command.add_argument("--dnsmos-model", help="optional local sig_bak_ovr.onnx")
    selection = commands.add_parser("select")
    selection.add_argument("database", type=Path)
    selection.add_argument("--policy", type=Path, required=True)
    selection.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "select":
            result = select(args.database, args.policy, args.out)
        else:
            from .backends import SileroVAD, WhisperASR

            backend = WhisperASR if args.command == "score" else SileroVAD
            model = backend(args.model, args.revision, local_files_only=args.offline)
            if args.command == "prepare":
                result = {"segments": prepare(args.manifest, args.out, model)}
            else:
                quality = None
                if args.dnsmos_model:
                    from .quality import DNSMOS

                    quality = DNSMOS(args.dnsmos_model)
                result = score(args.manifest, args.out, model, quality)
        print(json.dumps(result, ensure_ascii=False))
        if result.get("errors", 0) or result.get("error", 0):
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as e:
        parser.exit(2, f"error: {e}\n")


if __name__ == "__main__":
    main()
