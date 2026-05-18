"""共享工具：加载配置、客户端、路径。"""
from __future__ import annotations
import os, json, yaml
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def doubao_client(cfg: dict | None = None) -> OpenAI:
    cfg = cfg or load_config()
    return OpenAI(
        api_key=cfg["api_keys"]["doubao_ark"],
        base_url=cfg["api_keys"]["doubao_base_url"],
    )


def project_dir(sub: str = "") -> Path:
    cfg = load_config()
    p = ROOT / cfg["output"]["project_root"].lstrip("./")
    if sub:
        p = p / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_outline() -> dict:
    path = project_dir("01_script") / "outline.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_characters() -> dict:
    return load_outline().get("characters", {})


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
