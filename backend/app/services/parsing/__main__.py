import argparse
import asyncio
import fcntl
import json
import os
import sys
from pathlib import Path

from app.core.config import Settings
from app.services.parsing.errors import ParsingError
from app.services.parsing.pipeline import parse_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="显式调用 MinerU 云解析，会上传 PDF 并消耗解析额度"
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="本任务的私有工作目录")
    parser.add_argument("--batch-id", help="核对远端任务后恢复已有批次，不重新提交")
    parser.add_argument("--no-llm", action="store_true", help="跳过快速模型，仍查询 Crossref/arXiv")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.chmod(0o700)
    with (args.output / ".lock").open("w") as lock:
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("该解析任务正在运行", file=sys.stderr)
            sys.exit(1)
        try:
            report = asyncio.run(
                parse_pdf(
                    args.pdf,
                    args.output,
                    Settings(),
                    batch_id=args.batch_id,
                    use_llm=not args.no_llm,
                    notify=lambda s: print(s, file=sys.stderr),
                )
            )
        except ParsingError as error:
            print(
                json.dumps({"error": error.code, "message": str(error)}, ensure_ascii=False),
                file=sys.stderr,
            )
            sys.exit(1)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
