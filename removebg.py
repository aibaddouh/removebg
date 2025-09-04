#!/usr/bin/env python3
"""
removebg.py

Remove backgrounds from one or more images using rembg.

First run will download and install required dependencies in the user's home directory. Subsequent
runs will reuse the environment.

Usage examples:
  - GUI mode:
      ./removebg.py

  - Remove background from a single file:
      ./removebg.py /path/to/image.jpg

  - Multiple files and URLs, save to custom folder:
      ./removebg.py img1.jpg https://example.com/cat.png --outdir ./out

  - Overwrite existing outputs:
      ./removebg.py img1.jpg img2.png --overwrite
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import os
import re
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import platform


VENV_DIR_NAME = ".removebg_venv"
ENV_FLAG = "REMOVE_BG_BOOTSTRAPPED"
REQUIRED_PACKAGES = [
    "rembg>=2.0.0",
    "pillow>=10.0.0",
    "requests>=2.31.0",
]
ORT_MIN_VERSION = "1.17.0"


def get_venv_path() -> Path:
    home = Path.home()
    return home / VENV_DIR_NAME


def get_venv_binaries(venv_path: Path) -> Tuple[Path, Path]:
    if sys.platform.startswith("win"):
        bin_dir = venv_path / "Scripts"
        py = bin_dir / "python.exe"
        pip = bin_dir / "pip.exe"
    else:
        bin_dir = venv_path / "bin"
        py = bin_dir / "python3"
        if not py.exists():
            py = bin_dir / "python"
        pip = bin_dir / "pip"
    return py, pip


def create_venv(venv_path: Path) -> None:
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=True, upgrade=False)
    builder.create(str(venv_path))


def run_cmd(cmd: Sequence[str]) -> None:
    completed = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def ensure_deps_installed(python_bin: Path, pip_bin: Path) -> None:
    run_cmd([str(pip_bin), "install", "--upgrade", "pip", "setuptools", "wheel"])
    run_cmd([str(pip_bin), "install", "--upgrade", *REQUIRED_PACKAGES])

    # Ensure onnxruntime is available (rembg imports it at runtime)
    def has_ort() -> bool:
        try:
            subprocess.check_call([str(python_bin), "-c", "import onnxruntime"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False

    if not has_ort():
        system = platform.system().lower()
        machine = platform.machine().lower()
        ort_candidates: List[str] = []
        if system == "darwin" and machine in ("arm64", "aarch64"):
            ort_candidates = [
                f"onnxruntime-silicon>={ORT_MIN_VERSION}",
                f"onnxruntime>={ORT_MIN_VERSION}",
            ]
        else:
            ort_candidates = [f"onnxruntime>={ORT_MIN_VERSION}"]

        last_error: Optional[Exception] = None
        for pkg in ort_candidates:
            try:
                print(f"Installing {pkg}…")
                run_cmd([str(pip_bin), "install", "--upgrade", pkg])
                if has_ort():
                    last_error = None
                    break
            except Exception as exc:
                last_error = exc

        if not has_ort():
            raise RuntimeError(
                "Failed to install onnxruntime. Please install it manually inside the venv: "
                f"{pip_bin} install onnxruntime or onnxruntime-silicon. Last error: {last_error}"
            )

    check_code = (
        "import importlib;"
        "importlib.import_module('rembg');"
        "importlib.import_module('PIL');"
        "importlib.import_module('requests');"
        "importlib.import_module('onnxruntime')"
    )
    run_cmd([str(python_bin), "-c", check_code])


def in_our_venv(venv_path: Path) -> bool:
    try:
        return Path(sys.prefix).resolve() == venv_path.resolve()
    except Exception:
        return False


def bootstrap_and_reexec_if_needed() -> None:
    venv_path = get_venv_path()
    already_bootstrapped = os.environ.get(ENV_FLAG) == "1"

    if in_our_venv(venv_path) and already_bootstrapped:
        return

    if not venv_path.exists():
        print(f"[removebg] Creating private environment at {venv_path}")
        create_venv(venv_path)

    python_bin, pip_bin = get_venv_binaries(venv_path)

    need_install = False
    try:
        code = (
            "import importlib;"
            "import sys;"
            "importlib.import_module('rembg');"
            "importlib.import_module('PIL');"
            "importlib.import_module('requests')"
        )
        subprocess.check_call([str(python_bin), "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        need_install = True

    if need_install:
        print("Installing dependencies… this may take a minute the first time")
        ensure_deps_installed(python_bin, pip_bin)

    if not in_our_venv(venv_path):
        env = os.environ.copy()
        env[ENV_FLAG] = "1"
        abs_self = str(Path(__file__).resolve())
        args = [str(python_bin), abs_self, *sys.argv[1:]]
        os.execve(str(python_bin), args, env)


bootstrap_and_reexec_if_needed()


from PIL import Image
import requests
from rembg import remove, new_session


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Task:
    source: str
    is_url: bool
    output_path: Path


def is_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")


def sanitize_filename_from_url(url: str) -> str:
    from urllib.parse import urlparse, unquote

    parsed = urlparse(url)
    path = parsed.path
    name = Path(unquote(path)).name or "download"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not Path(name).suffix:
        name = f"{name}.png"
    return name


def discover_files(arg: str) -> List[Path]:
    p = Path(arg)
    if p.exists():
        if p.is_file():
            return [p]
        if p.is_dir():
            results: List[Path] = []
            for ext in IMAGE_EXTS:
                results.extend(p.rglob(f"*{ext}"))
            return results
    matches = list(Path().glob(arg))
    return [m for m in matches if m.is_file()]


def build_tasks(inputs: Sequence[str], outdir: Optional[str], suffix: str) -> List[Task]:
    tasks: List[Task] = []
    out_base = Path(outdir) if outdir else None

    for item in inputs:
        if is_url(item):
            base_name = sanitize_filename_from_url(item)
            base_stem = Path(base_name).stem
            out_dir = out_base or Path.cwd()
            out_path = out_dir / f"{base_stem}{suffix}.png"
            tasks.append(Task(source=item, is_url=True, output_path=out_path))
            continue

        files = discover_files(item)
        for f in files:
            out_dir = out_base or f.parent
            out_path = out_dir / f"{f.stem}{suffix}.png"
            tasks.append(Task(source=str(f), is_url=False, output_path=out_path))

    dedup: dict[Path, Task] = {}
    for t in tasks:
        dedup[t.output_path] = t
    return list(dedup.values())


def load_image(task: Task) -> Image.Image:
    if task.is_url:
        resp = requests.get(task.source, timeout=60)
        resp.raise_for_status()
        data = io.BytesIO(resp.content)
        img = Image.open(data)
    else:
        img = Image.open(task.source)
    # Ensure RGBA for transparent output
    return img.convert("RGBA")


def process_image_pil(img: Image.Image, session=None) -> Image.Image:
    """Process a single PIL image and return the result with background removed."""
    if session is None:
        session = new_session("isnet-general-use")
    return remove(
        img.convert("RGBA"),
        post_process_mask=True,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=270,
        alpha_matting_background_threshold=20,
        alpha_matting_erode_size=11,
    )


def process_one(task: Task, overwrite: bool) -> Tuple[Task, Optional[str]]:
    try:
        # rembg uses a small UNet model to segment subject from background
        session = new_session("isnet-general-use")
        if task.output_path.exists() and not overwrite:
            return task, f"exists (skip): {task.output_path}"

        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        img = load_image(task)

        # Perform background removal
        out_img = process_image_pil(img, session=session)
        out_img.save(task.output_path, format="PNG")
        return task, None
    except Exception as exc:
        return task, f"error: {exc}"


def launch_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog
        from PIL import ImageTk
    except Exception as exc:
        print("Tkinter GUI is unavailable in this Python. Please run via CLI or install Python with Tk support.")
        print(f"Reason: {exc}")
        return 1

    @dataclass
    class GUIItem:
        source: str
        is_url: bool
        original: Optional[Image.Image] = None
        processed: Optional[Image.Image] = None

    class App:
        def __init__(self, root: "tk.Tk") -> None:
            self.root = root
            self.root.title("removebg - Preview and Save")
            self.items: List[GUIItem] = []
            self.session = None

            self._build_ui()

        def _build_ui(self) -> None:
            toolbar = tk.Frame(self.root)
            toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

            tk.Button(toolbar, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=4)
            tk.Button(toolbar, text="Add Folder", command=self.add_folder).pack(side=tk.LEFT, padx=4)
            tk.Button(toolbar, text="Add URL", command=self.add_url).pack(side=tk.LEFT, padx=4)
            tk.Button(toolbar, text="Process Selected", command=self.process_selected).pack(side=tk.LEFT, padx=12)
            tk.Button(toolbar, text="Process All", command=self.process_all).pack(side=tk.LEFT, padx=4)
            tk.Button(toolbar, text="Save", command=self.save_selected).pack(side=tk.RIGHT, padx=4)
            tk.Button(toolbar, text="Save All", command=self.save_all).pack(side=tk.RIGHT, padx=4)

            body = tk.Frame(self.root)
            body.pack(fill=tk.BOTH, expand=True)

            left = tk.Frame(body, width=240)
            left.pack(side=tk.LEFT, fill=tk.Y)
            right = tk.Frame(body)
            right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            self.listbox = tk.Listbox(left, selectmode=tk.EXTENDED)
            self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
            self.listbox.bind("<<ListboxSelect>>", lambda e: self.update_previews())

            preview_frame = tk.Frame(right)
            preview_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

            titles = tk.Frame(preview_frame)
            titles.pack(fill=tk.X)
            tk.Label(titles, text="Original").pack(side=tk.LEFT, padx=4)
            tk.Label(titles, text="Processed").pack(side=tk.RIGHT, padx=4)

            images_frame = tk.Frame(preview_frame)
            images_frame.pack(fill=tk.BOTH, expand=True)

            self.orig_label = tk.Label(images_frame, bg="#f0f0f0", width=40, height=20)
            self.orig_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
            self.proc_label = tk.Label(images_frame, bg="#f8f8f8", width=40, height=20)
            self.proc_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)

            status = tk.Frame(self.root)
            status.pack(fill=tk.X)
            self.status_var = tk.StringVar(value="Ready.")
            self.status_label = tk.Label(status, textvariable=self.status_var, anchor="w")
            self.status_label.pack(fill=tk.X, padx=8, pady=4)

            self._orig_imgtk = None
            self._proc_imgtk = None

        def set_status(self, text: str) -> None:
            self.status_var.set(text)
            self.root.update_idletasks()

        def add_files(self) -> None:
            filetypes = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")]
            paths = filedialog.askopenfilenames(title="Select images", filetypes=filetypes)
            for p in paths:
                self._append_item(str(p), is_url=False)

        def add_folder(self) -> None:
            d = filedialog.askdirectory(title="Select folder")
            if not d:
                return
            for ext in IMAGE_EXTS:
                for p in Path(d).rglob(f"*{ext}"):
                    self._append_item(str(p), is_url=False)

        def add_url(self) -> None:
            url = simpledialog.askstring("Add URL", "Enter image URL:")
            if not url:
                return
            if not is_url(url):
                messagebox.showerror("Invalid URL", "Please enter a valid http(s) URL.")
                return
            self._append_item(url, is_url=True)

        def _append_item(self, source: str, is_url: bool) -> None:
            self.items.append(GUIItem(source=source, is_url=is_url))
            name = Path(source).name if not is_url else source
            self.listbox.insert(tk.END, name)
            if self.listbox.size() == 1:
                self.listbox.selection_set(0)
                self.update_previews()

        def _load_original(self, item: GUIItem) -> Optional[Image.Image]:
            if item.original is not None:
                return item.original
            try:
                if item.is_url:
                    resp = requests.get(item.source, timeout=60)
                    resp.raise_for_status()
                    data = io.BytesIO(resp.content)
                    img = Image.open(data)
                else:
                    img = Image.open(item.source)
                item.original = img.convert("RGBA")
                return item.original
            except Exception as exc:
                self.set_status(f"Failed to load: {exc}")
                return None

        def update_previews(self) -> None:
            sel = self._current_index()
            if sel is None:
                self._set_preview_images(None, None)
                return
            item = self.items[sel]
            orig = self._load_original(item)
            proc = item.processed
            self._set_preview_images(orig, proc)

        def _thumbnail(self, img: Optional[Image.Image], max_size: Tuple[int, int]) -> Optional[Image.Image]:
            if img is None:
                return None
            copy = img.copy()
            copy.thumbnail(max_size, Image.LANCZOS)
            return copy

        def _set_preview_images(self, orig: Optional[Image.Image], proc: Optional[Image.Image]) -> None:
            w1 = self.orig_label.winfo_width() or 400
            h1 = self.orig_label.winfo_height() or 400
            w2 = self.proc_label.winfo_width() or 400
            h2 = self.proc_label.winfo_height() or 400
            o_thumb = self._thumbnail(orig, (w1, h1)) if orig else None
            p_thumb = self._thumbnail(proc, (w2, h2)) if proc else None

            if o_thumb is not None:
                self._orig_imgtk = ImageTk.PhotoImage(o_thumb)
                self.orig_label.configure(image=self._orig_imgtk)
            else:
                self.orig_label.configure(image="")
                self._orig_imgtk = None

            if p_thumb is not None:
                self._proc_imgtk = ImageTk.PhotoImage(p_thumb)
                self.proc_label.configure(image=self._proc_imgtk)
            else:
                self.proc_label.configure(image="")
                self._proc_imgtk = None

        def _current_index(self) -> Optional[int]:
            sel = self.listbox.curselection()
            if not sel:
                return None
            return int(sel[0])

        def _ensure_session(self) -> None:
            if self.session is None:
                self.set_status("Loading model…")
                self.root.update_idletasks()
                self.session = new_session("isnet-general-use")

        def process_selected(self) -> None:
            selection = list(self.listbox.curselection())
            if not selection:
                messagebox.showinfo("Process", "Select one or more items in the list to process.")
                return
            self._ensure_session()
            for idx in selection:
                self._process_index(idx)
            self.update_previews()
            self.set_status("Processing complete.")

        def process_all(self) -> None:
            if not self.items:
                return
            self._ensure_session()
            for idx in range(len(self.items)):
                self._process_index(idx)
            self.update_previews()
            self.set_status("Processing complete.")

        def _process_index(self, idx: int) -> None:
            try:
                item = self.items[idx]
                orig = self._load_original(item)
                if orig is None:
                    return
                out_img = process_image_pil(orig, session=self.session)
                item.processed = out_img
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to process: {exc}")

        def save_selected(self) -> None:
            idx = self._current_index()
            if idx is None:
                return
            item = self.items[idx]
            if item.processed is None:
                messagebox.showinfo("Save", "Process the image first.")
                return
            base = Path(sanitize_filename_from_url(item.source) if item.is_url else item.source)
            default = f"{base.stem}_nobg.png"
            path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default, filetypes=[("PNG", "*.png")])
            if not path:
                return
            try:
                item.processed.save(path, format="PNG")
                self.set_status(f"Saved: {path}")
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))

        def save_all(self) -> None:
            if not self.items:
                return
            any_processed = any(i.processed is not None for i in self.items)
            if not any_processed:
                messagebox.showinfo("Save All", "No processed images to save. Process first.")
                return
            folder = filedialog.askdirectory(title="Choose output folder")
            if not folder:
                return
            for item in self.items:
                if item.processed is None:
                    continue
                base = Path(sanitize_filename_from_url(item.source) if item.is_url else item.source)
                out_path = Path(folder) / f"{base.stem}_nobg.png"
                try:
                    item.processed.save(out_path, format="PNG")
                except Exception as exc:
                    messagebox.showerror("Save failed", f"{out_path}: {exc}")
            self.set_status(f"Saved to: {folder}")

    root = tk.Tk()
    root.geometry("1000x650")
    _ = App(root)
    root.mainloop()
    return 0

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="removebg",
        description="Remove image backgrounds using rembg. Accepts files, directories, globs, or URLs.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="One or more file paths, directories, globs, or URLs",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to each input's folder (files) or CWD (URLs).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_nobg",
        help="Suffix appended to base filename before .png (default: _nobg)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (default: skip existing)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(32, (os.cpu_count() or 2)),
        help="Number of parallel workers (default: number of CPUs, up to 32)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Launch GUI mode when no inputs and no flags were provided
    if not args.inputs and len(sys.argv) == 1:
        return launch_gui()

    tasks = build_tasks(args.inputs, args.outdir, args.suffix)
    if not tasks:
        print("No images found for given inputs.")
        return 1

    print(f"Processing {len(tasks)} image(s)…")
    failures = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        futures = [executor.submit(process_one, task, args.overwrite) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            task, error = fut.result()
            if error is None:
                print(f"  ✓ {task.output_path}")
            else:
                print(f"  ✗ {task.source} -> {error}")
                failures += 1

    if failures:
        print(f"Completed with {failures} failure(s).")
        return 2

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)


