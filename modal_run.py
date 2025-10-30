"""BoltzGen fork https://github.com/y1zhou/boltzgen.

Example usage:
```bash
uvx modal run modal_run.py \
    --input-yaml example/nanobody_against_penguinpox/penguinpox.yaml \
    --protocol nanobody-anything \
    --num-designs 1 \
    --out-dir ./modal_output
```
"""
# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603

import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "120"))

BOLTZGEN_VOLUME_NAME = "boltzgen-models"
BOLTZGEN_MODEL_VOLUME = Volume.from_name(BOLTZGEN_VOLUME_NAME, create_if_missing=True)
CACHE_DIR = f"/{BOLTZGEN_VOLUME_NAME}"
REPO_DIR = "/root/boltzgen"


def download_boltzgen_models() -> None:
    """Download all boltzgen models during image build to avoid runtime timeouts."""
    import shutil
    import subprocess

    uv_exe = shutil.which("uv")
    if uv_exe is None:
        raise RuntimeError("uv executable not found on PATH")

    # Download all artifacts (~/.cache overridden to volume mount)
    print("Downloading boltzgen models...")
    subprocess.run(
        [uv_exe, "run", "boltzgen", "download", "all", "--cache", CACHE_DIR],
        check=True,
        cwd=REPO_DIR,
    )
    print("Model download complete")


image = (
    Image.debian_slim()
    .apt_install("git", "wget", "build-essential")
    .run_commands(
        f"git clone https://github.com/HannesStark/boltzgen {REPO_DIR}",
        f"cd {REPO_DIR} && "
        "git checkout 894d8eb3069de696f26a6f3dc801685a7be0a791 && "
        "uv venv --python 3.12 && "
        "uv pip install -e . --torch-backend=auto",
        gpu="a10g",
    )
    .workdir(REPO_DIR)
    .run_function(download_boltzgen_models, volumes={CACHE_DIR: BOLTZGEN_MODEL_VOLUME})
)

app = App("BoltzGen", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def boltzgen_run(
    yaml_str: str,
    yaml_name: str,
    additional_files: dict[str, bytes],
    protocol: str = "protein-anything",
    num_designs: int = 10,
    steps: str | None = None,
    cache: str | None = CACHE_DIR,
    devices: int | None = None,
    extra_args: str | None = None,
) -> list:
    """Run BoltzGen on a yaml specification.

    Args:
        yaml_str: YAML design specification as string
        yaml_name: Name of the yaml file
        additional_files: Dict of relative_path -> file_content for referenced files
        protocol: Design protocol (protein-anything, peptide-anything, etc.)
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        cache: Custom cache directory path
        devices: Number of GPUs to use
        extra_args: Additional CLI arguments as string

    Returns
    -------
        List of (path, content) tuples for all output files
    """
    import shutil
    from subprocess import run
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        # Write yaml to file
        in_path = Path(in_dir).resolve() / "config"
        in_path.mkdir(parents=True, exist_ok=True)
        yaml_path = in_path / yaml_name
        yaml_path.write_text(yaml_str)

        # Write any additional files (e.g., .cif files referenced in yaml)
        for rel_path, content in additional_files.items():
            file_path = in_path / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)

        # Build command
        uv_exe = shutil.which("uv")
        if uv_exe is None:
            raise RuntimeError("uv executable not found on PATH")
        cmd = [
            uv_exe,
            "run",
            "boltzgen",
            "run",
            str(yaml_path),
            "--protocol",
            protocol,
            "--output",
            out_dir,
            "--num_designs",
            str(num_designs),
        ]

        if steps:
            cmd.extend(["--steps", *steps.split()])
        if cache:
            cmd.extend(["--cache", cache])
        if devices:
            cmd.extend(["--devices", str(devices)])
        if extra_args:
            cmd.extend(extra_args.split())

        print(f"Running: {' '.join(cmd)}")
        run(cmd, check=True, cwd=REPO_DIR)

        # Collect all output files
        return [
            (out_file.relative_to(out_dir), out_file.read_bytes())
            for out_file in Path(out_dir).rglob("*")
            if out_file.is_file()
        ]


class YAMLReferenceLoader:
    """Class to load referenced files from YAML files.

    BoltzGen configs might reference other cif or yaml files.
    We need to recursively parse all yaml files to find all used cif templates.

    The file paths need to be relative to the parent directory of the
    input yaml, because we need to recreate the file structure on the remote.
    """

    def __init__(self, input_yaml_file: str | Path) -> None:
        self.input_path = Path(input_yaml_file).expanduser().resolve()
        self.ref_dir = self.input_path.parent

        # key: relative path to self.ref_dir, value: file content bytes
        self.additional_files: dict[str, bytes] = {}

        # absolute paths for tracking and recursive parsing
        self.parsed_files: set[Path] = set()
        self.queue: set[Path] = set()
        self.queue.add(self.input_path)
        self.load()

    def load(self) -> None:
        """Load referenced files from a YAML."""
        while self.queue:
            file = self.queue.pop()
            if file in self.parsed_files:
                continue

            new_ref_files = self.find_paths_from_yaml(file)
            for ref_file in new_ref_files:
                ref_path = file.parent.joinpath(ref_file).resolve()
                if ref_path.exists():
                    rel_path = ref_path.relative_to(self.ref_dir, walk_up=True)
                    self.additional_files[str(rel_path)] = ref_path.read_bytes()
                if (
                    ref_path.suffix in {".yaml", ".yml"}
                    and ref_path not in self.parsed_files
                ):
                    self.queue.add(ref_path)

    def find_paths_from_yaml(self, yaml_file: Path) -> set[Path]:
        """Load referenced files from a YAML."""
        import yaml

        yaml_path = Path(yaml_file).expanduser().resolve()
        if yaml_path in self.parsed_files:
            return set()

        with yaml_path.open() as f:
            conf = yaml.safe_load(f)

        file_refs: set[Path] = set()
        self.find_paths_in_dict(conf, yaml_path.parent, file_refs)
        self.parsed_files.add(yaml_path)
        return file_refs

    def find_paths_in_dict(
        self, yaml_content: dict, ref_dir: Path, file_refs: set[Path]
    ) -> None:
        """Recursively find all file references in the yaml content."""
        for v in yaml_content.values():
            if isinstance(v, str):
                if (p := (ref_dir / v)).exists():
                    file_refs.add(p)
            elif isinstance(v, list):
                self.find_paths_in_list(v, ref_dir, file_refs)
            elif isinstance(v, dict):
                self.find_paths_in_dict(v, ref_dir, file_refs)
            else:
                continue

    def find_paths_in_list(
        self, sublist: list, ref_dir: Path, file_refs: set[Path]
    ) -> None:
        """Recursively find all file references in the yaml content."""
        for item in sublist:
            if isinstance(item, str):
                if (p := (ref_dir / item)).exists():
                    file_refs.add(p)
            elif isinstance(item, dict):
                self.find_paths_in_dict(item, ref_dir, file_refs)
            elif isinstance(item, list):
                self.find_paths_in_list(item, ref_dir, file_refs)
            else:
                continue


@app.local_entrypoint()
def main(
    input_yaml: str,
    protocol: str = "protein-anything",
    num_designs: int = 10,
    steps: str | None = None,
    cache: str | None = None,
    devices: int | None = None,
    extra_args: str | None = None,
    out_dir: str = "./out/boltzgen",
    run_name: str | None = None,
) -> None:
    """Run BoltzGen locally with results saved to out_dir.

    Args:
        input_yaml: Path to YAML design specification file
        protocol: Design protocol, one of: protein-anything, peptide-anything,
            protein-small_molecule, or nanobody-anything
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        cache: Custom cache directory path
        devices: Number of GPUs to use
        extra_args: Additional CLI arguments as string
        out_dir: Local output directory
        run_name: Optional run name (defaults to timestamp)
    """
    from datetime import datetime, timezone

    yaml_path = Path(input_yaml)
    yaml_str = yaml_path.read_text()

    # Find any file references in the yaml (path: something.cif)
    # File paths in yaml are relative to the yaml file location
    yml_parser = YAMLReferenceLoader(yaml_path)
    print(
        f"Including additional referenced files: {list(yml_parser.additional_files.keys())}"
    )

    outputs = boltzgen_run.remote(
        yaml_str=yaml_str,
        yaml_name=yaml_path.name,
        additional_files=yml_parser.additional_files,
        protocol=protocol,
        num_designs=num_designs,
        steps=steps,
        cache=cache,
        devices=devices,
        extra_args=extra_args,
    )

    today = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        output_path = out_dir_full / out_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(out_content)

    print(f"\nResults saved to: {out_dir_full}")
