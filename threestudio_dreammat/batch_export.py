import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf
from PIL import Image


REQUIRED_OUTPUT_FILES = ("albedo.png", "metallic.png", "roughness.png", "mesh.obj")
FILTERED_EXTRA_PREFIXES = (
    "exp_root_dir=",
    "name=",
    "tag=",
    "system.prompt_processor.prompt=",
    "system.geometry.shape_init=",
    "system.exporter.fmt=",
    "system.exporter.save_name=",
)


def maybe_run_batch(args, extras) -> bool:
    if not getattr(args, "batch_tsv", None):
        return False
    run_batch_export(args, extras)
    return True


def run_batch_export(args, extras) -> None:
    if not args.train:
        raise ValueError("Batch export currently requires --train.")

    tsv_path = os.path.abspath(args.batch_tsv)
    rows = load_batch_from_tsv(tsv_path, args.caption_field)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    requested_gpu_ids = args.gpu_ids or args.gpu
    gpu_ids = parse_gpu_ids(requested_gpu_ids)
    if not gpu_ids:
        raise ValueError("No valid GPU IDs provided for batch export.")
    if args.num_gpus > 0:
        gpu_ids = gpu_ids[: min(args.num_gpus, len(gpu_ids))]
    workers_per_gpu = parse_workers_per_gpu(args.workers_per_gpu)
    worker_slots = build_worker_slots(gpu_ids, workers_per_gpu)

    batch_root = resolve_batch_root(args.config, extras)
    textures_dir = os.path.join(batch_root, args.batch_textures_dir_name)
    result_tsv_path = resolve_result_tsv(batch_root, args.result_tsv)
    os.makedirs(textures_dir, exist_ok=True)

    filtered_extras = filter_forwarded_extras(extras)
    options = build_options_payload(
        args=args,
        filtered_extras=filtered_extras,
        batch_root=batch_root,
        tsv_path=tsv_path,
        result_tsv_path=result_tsv_path,
        effective_gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
        total_workers=len(worker_slots),
    )

    launch_script = os.path.abspath(sys.argv[0])
    launch_cwd = os.getcwd()
    tsv_dir = os.path.dirname(tsv_path)
    processed_samples: List[Dict[str, Any]] = []
    skipped_samples: List[Dict[str, Any]] = []
    pending_tasks: List[Dict[str, Any]] = []
    auto_skipped_count = 0

    start_ts = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    processing_start_ts: Optional[float] = None
    processing_start_dt: Optional[str] = None
    processing_end_ts: Optional[float] = None
    processing_end_dt: Optional[str] = None
    print(f"[INFO] Loaded {len(rows)} rows from {tsv_path}")
    print(f"[INFO] Batch output directory: {batch_root}")
    print(f"[INFO] Using caption field: {args.caption_field}")
    print(f"[INFO] Using GPU IDs: {gpu_ids}")
    print(f"[INFO] Workers per GPU: {workers_per_gpu}")
    print(f"[INFO] Total parallel workers: {len(worker_slots)}")

    try:
        total_rows = len(rows)
        for row_index, row in enumerate(rows, start=1):
            obj_id = clean_field(row.get("obj_id"))
            if not obj_id:
                skipped_samples.append(
                    {"obj_id": "", "error": "Missing obj_id in TSV row", "row_index": row_index}
                )
                continue

            try:
                mesh_path = resolve_mesh_path(row, tsv_dir)
            except Exception as exc:
                skipped_samples.append({"obj_id": obj_id, "error": str(exc)})
                print(f"[WARN] [{row_index}/{total_rows}] Skip {obj_id}: {exc}")
                continue

            sample_output_dir = os.path.join(textures_dir, obj_id)
            if check_sample_completed(sample_output_dir, mesh_path):
                auto_skipped_count += 1
                print(f"[INFO] [{row_index}/{total_rows}] Auto-skip completed sample {obj_id}")
                completed_row = build_result_row(
                    obj_id=obj_id,
                    sample_dir=sample_output_dir,
                    caption_short=clean_field(row.get("caption_short")),
                    caption_long=clean_field(row.get("caption_long")),
                    caption_used=args.caption_field,
                )
                processed_samples.append(completed_row)
                append_to_manifest(result_tsv_path, [completed_row])
                continue

            prompt = clean_field(row.get(args.caption_field))
            if not prompt:
                skipped_samples.append(
                    {
                        "obj_id": obj_id,
                        "error": f"Missing prompt text in column '{args.caption_field}'",
                    }
                )
                print(
                    f"[WARN] [{row_index}/{total_rows}] Skip {obj_id}: missing "
                    f"'{args.caption_field}'"
                )
                continue

            pending_tasks.append(
                {
                    "row_index": row_index,
                    "total_rows": total_rows,
                    "obj_id": obj_id,
                    "mesh_path": mesh_path,
                    "prompt": prompt,
                    "sample_output_dir": sample_output_dir,
                    "trial_dir": os.path.join(batch_root, args.batch_work_dir_name, obj_id),
                    "caption_short": clean_field(row.get("caption_short")),
                    "caption_long": clean_field(row.get("caption_long")),
                }
            )

        if auto_skipped_count:
            print(f"[INFO] Auto-skipped {auto_skipped_count} already-completed samples")
        print(f"[INFO] Remaining samples to process: {len(pending_tasks)}")

        if pending_tasks:
            processing_start_ts = time.time()
            processing_start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_processed, new_skipped = run_pending_tasks(
                pending_tasks=pending_tasks,
                worker_slots=worker_slots,
                args=args,
                filtered_extras=filtered_extras,
                launch_script=launch_script,
                launch_cwd=launch_cwd,
                batch_root=batch_root,
                result_tsv_path=result_tsv_path,
            )
            processing_end_ts = time.time()
            processing_end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            processed_samples.extend(new_processed)
            skipped_samples.extend(new_skipped)
        else:
            print("[INFO] No remaining samples to process.")
    except KeyboardInterrupt:
        print("[WARN] Batch export interrupted by user.")
    finally:
        if processing_start_ts is None:
            effective_start_ts = start_ts
            effective_start_dt = start_dt
        else:
            effective_start_ts = processing_start_ts
            effective_start_dt = processing_start_dt or start_dt

        if processing_end_ts is None:
            effective_end_ts = time.time()
            effective_end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            effective_end_ts = processing_end_ts
            effective_end_dt = processing_end_dt or datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        total_seconds = effective_end_ts - effective_start_ts
        num_processed = len(processed_samples)
        timing_info = {
            "start_time": effective_start_dt,
            "end_time": effective_end_dt,
            "total_seconds": round(total_seconds, 2),
            "total_time_formatted": str(timedelta(seconds=int(total_seconds))),
            "num_samples_processed": num_processed,
            "avg_seconds_per_sample": round(total_seconds / num_processed, 2)
            if num_processed
            else 0.0,
            "gpu_ids": gpu_ids,
            "num_gpus": len(gpu_ids),
            "workers_per_gpu": workers_per_gpu,
            "total_workers": len(worker_slots),
        }
        merge_and_save_batch_config(
            exp_dir=batch_root,
            options=options,
            new_processed=processed_samples,
            new_skipped=skipped_samples,
            manifest_path=result_tsv_path if os.path.isfile(result_tsv_path) else None,
            timing_info=timing_info,
        )
        cleanup_empty_dir(os.path.join(batch_root, args.batch_work_dir_name))

        print("\n" + "=" * 60)
        print("[TIMING] Batch export completed")
        print(f"[TIMING] Start time: {effective_start_dt}")
        print(f"[TIMING] End time: {effective_end_dt}")
        print(
            f"[TIMING] Total time: {timing_info['total_time_formatted']} "
            f"({timing_info['total_seconds']:.2f} seconds)"
        )
        print(f"[TIMING] Samples processed: {num_processed}")
        print(f"[TIMING] Samples skipped: {len(skipped_samples)}")
        print(f"[TIMING] GPU IDs: {gpu_ids}")
        print(f"[TIMING] Workers per GPU: {workers_per_gpu}")
        print("=" * 60 + "\n")


def run_pending_tasks(
    pending_tasks: List[Dict[str, Any]],
    worker_slots: List[Dict[str, int]],
    args,
    filtered_extras: List[str],
    launch_script: str,
    launch_cwd: str,
    batch_root: str,
    result_tsv_path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    processed_samples: List[Dict[str, Any]] = []
    skipped_samples: List[Dict[str, Any]] = []

    if len(worker_slots) <= 1 or len(pending_tasks) <= 1:
        slot = worker_slots[0]
        for task in pending_tasks:
            result = process_batch_sample(
                task=task,
                worker_slot=slot,
                args=args,
                filtered_extras=filtered_extras,
                launch_script=launch_script,
                launch_cwd=launch_cwd,
                batch_root=batch_root,
            )
            consume_worker_result(
                result=result,
                processed_samples=processed_samples,
                skipped_samples=skipped_samples,
                result_tsv_path=result_tsv_path,
            )
        return processed_samples, skipped_samples

    task_iter = iter(pending_tasks)
    future_to_slot = {}

    with ThreadPoolExecutor(max_workers=len(worker_slots)) as executor:
        for slot in worker_slots:
            task = next_task_or_none(task_iter)
            if task is None:
                break
            future = executor.submit(
                process_batch_sample,
                task,
                slot,
                args,
                filtered_extras,
                launch_script,
                launch_cwd,
                batch_root,
            )
            future_to_slot[future] = slot

        while future_to_slot:
            done, _ = wait(future_to_slot.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                slot = future_to_slot.pop(future)
                result = future.result()
                consume_worker_result(
                    result=result,
                    processed_samples=processed_samples,
                    skipped_samples=skipped_samples,
                    result_tsv_path=result_tsv_path,
                )
                task = next_task_or_none(task_iter)
                if task is None:
                    continue
                new_future = executor.submit(
                    process_batch_sample,
                    task,
                    slot,
                    args,
                    filtered_extras,
                    launch_script,
                    launch_cwd,
                    batch_root,
                )
                future_to_slot[new_future] = slot

    return processed_samples, skipped_samples


def next_task_or_none(task_iter):
    try:
        return next(task_iter)
    except StopIteration:
        return None


def process_batch_sample(
    task: Dict[str, Any],
    worker_slot: Dict[str, int],
    args,
    filtered_extras: List[str],
    launch_script: str,
    launch_cwd: str,
    batch_root: str,
) -> Dict[str, Any]:
    obj_id = task["obj_id"]
    gpu_id = worker_slot["gpu_id"]
    worker_id = worker_slot["worker_id"]
    sample_output_dir = task["sample_output_dir"]
    trial_dir = task["trial_dir"]

    try:
        if os.path.isdir(sample_output_dir):
            shutil.rmtree(sample_output_dir)
        if os.path.isdir(trial_dir):
            shutil.rmtree(trial_dir)

        print(
            f"[INFO] [{task['row_index']}/{task['total_rows']}] "
            f"Worker {worker_id} on GPU {gpu_id} processing {obj_id}"
        )

        child_cmd = build_child_command(
            launch_script=launch_script,
            args=args,
            filtered_extras=filtered_extras,
            batch_root=batch_root,
            obj_id=obj_id,
            prompt=task["prompt"],
            mesh_path=task["mesh_path"],
            gpu_id=gpu_id,
        )
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        child_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        completed = subprocess.run(
            child_cmd,
            cwd=launch_cwd,
            env=child_env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Child launch exited with code {completed.returncode} for {obj_id}"
            )

        export_dir = find_latest_export_dir(trial_dir)
        canonicalize_exported_assets(export_dir, sample_output_dir)
        validate_sample_output(sample_output_dir)

        result_row = build_result_row(
            obj_id=obj_id,
            sample_dir=sample_output_dir,
            caption_short=task["caption_short"],
            caption_long=task["caption_long"],
            caption_used=args.caption_field,
        )

        if not args.keep_intermediate and os.path.isdir(trial_dir):
            shutil.rmtree(trial_dir)

        return {
            "status": "processed",
            "row_index": task["row_index"],
            "total_rows": task["total_rows"],
            "obj_id": obj_id,
            "gpu_id": gpu_id,
            "worker_id": worker_id,
            "result_row": result_row,
        }
    except Exception as exc:
        if os.path.isdir(sample_output_dir):
            shutil.rmtree(sample_output_dir)
        return {
            "status": "skipped",
            "row_index": task["row_index"],
            "total_rows": task["total_rows"],
            "obj_id": obj_id,
            "gpu_id": gpu_id,
            "worker_id": worker_id,
            "skip_info": {
                "obj_id": obj_id,
                "error": str(exc),
                "trial_dir": os.path.abspath(trial_dir),
                "gpu_id": gpu_id,
                "worker_id": worker_id,
            },
        }


def consume_worker_result(
    result: Dict[str, Any],
    processed_samples: List[Dict[str, Any]],
    skipped_samples: List[Dict[str, Any]],
    result_tsv_path: str,
) -> None:
    if result["status"] == "processed":
        processed_samples.append(result["result_row"])
        append_to_manifest(result_tsv_path, [result["result_row"]])
        print(
            f"[INFO] [{result['row_index']}/{result['total_rows']}] "
            f"Worker {result['worker_id']} on GPU {result['gpu_id']} finished "
            f"{result['obj_id']}"
        )
        return

    skipped_samples.append(result["skip_info"])
    print(
        f"[WARN] [{result['row_index']}/{result['total_rows']}] "
        f"Worker {result['worker_id']} on GPU {result['gpu_id']} failed "
        f"{result['obj_id']}: {result['skip_info']['error']}"
    )


def clean_field(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_batch_from_tsv(tsv_path: str, caption_field: str) -> List[Dict[str, str]]:
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        required = ["obj_id", "mesh", "caption_short", "caption_long"]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(
                "TSV missing required columns: "
                f"{', '.join(missing)} (available: {', '.join(fieldnames) or '(none)'})"
            )
        if caption_field not in fieldnames:
            raise ValueError(
                f"TSV missing caption_field '{caption_field}' "
                f"(available: {', '.join(fieldnames) or '(none)'})"
            )
        return [row for row in reader]


def resolve_batch_root(config_path: str, extras: List[str]) -> str:
    yaml_conf = OmegaConf.load(config_path)
    cli_conf = OmegaConf.from_cli(extras)
    merged = OmegaConf.merge(yaml_conf, cli_conf)
    OmegaConf.resolve(merged)
    exp_root_dir = os.path.abspath(str(merged.get("exp_root_dir", "outputs")))
    name = str(merged.get("name", "default"))
    return os.path.abspath(os.path.join(exp_root_dir, name))


def resolve_result_tsv(batch_root: str, result_tsv: Optional[str]) -> str:
    if not result_tsv:
        return os.path.abspath(os.path.join(batch_root, "generated_manifest.tsv"))
    if os.path.isabs(result_tsv):
        return os.path.abspath(result_tsv)
    return os.path.abspath(os.path.join(batch_root, result_tsv))


def filter_forwarded_extras(extras: List[str]) -> List[str]:
    filtered: List[str] = []
    for extra in extras:
        if any(extra.startswith(prefix) for prefix in FILTERED_EXTRA_PREFIXES):
            continue
        filtered.append(extra)
    return filtered


def parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    text = clean_field(gpu_ids_str)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_workers_per_gpu(value: Any) -> int:
    text = clean_field(value).lower()
    if text == "auto":
        print("[WARN] workers_per_gpu=auto is not implemented for DreamMat, fallback to 1")
        return 1
    parsed = int(text)
    if parsed <= 0:
        raise ValueError("workers_per_gpu must be a positive integer")
    return parsed


def build_worker_slots(gpu_ids: List[int], workers_per_gpu: int) -> List[Dict[str, int]]:
    slots: List[Dict[str, int]] = []
    worker_id = 0
    for gpu_id in gpu_ids:
        for local_worker_index in range(workers_per_gpu):
            slots.append(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "local_worker_index": local_worker_index,
                }
            )
            worker_id += 1
    return slots


def resolve_mesh_path(row: Dict[str, str], tsv_dir: str) -> str:
    mesh_path = clean_field(row.get("mesh") or row.get("mesh_path"))
    if not mesh_path:
        raise ValueError("Missing mesh path in TSV row")
    if mesh_path.startswith("mesh:"):
        mesh_path = mesh_path[len("mesh:") :]
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.abspath(os.path.join(tsv_dir, mesh_path))
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    return os.path.abspath(mesh_path)


def normalize_cli_string(value: str) -> str:
    return value.replace("\n", " ")


def build_child_command(
    launch_script: str,
    args,
    filtered_extras: List[str],
    batch_root: str,
    obj_id: str,
    prompt: str,
    mesh_path: str,
    gpu_id: int,
) -> List[str]:
    has_seed_override = any(extra.startswith("seed=") for extra in filtered_extras)
    fix_view_num = resolve_fix_view_num_override(filtered_extras)
    trial_dir = os.path.join(batch_root, args.batch_work_dir_name, obj_id)
    reuse_prerender = check_prerender_completed(trial_dir, fix_view_num)
    command = [
        sys.executable,
        launch_script,
        "--config",
        args.config,
        "--train",
        "--gradio",
        "--gpu",
        str(gpu_id),
    ]
    if args.verbose:
        command.append("--verbose")
    if args.typecheck:
        command.append("--typecheck")

    command.extend(filtered_extras)
    if not has_seed_override:
        command.append("seed=0")
    if reuse_prerender:
        command.append("data.blender_generate=false")
    command.extend(
        [
            f"exp_root_dir={batch_root}",
            f"name={args.batch_work_dir_name}",
            f"tag={obj_id}",
            f"system.prompt_processor.prompt={normalize_cli_string(prompt)}",
            f"system.geometry.shape_init=mesh:{mesh_path}",
            "system.geometry.export_coordinate_mode=texgaussian",
            "system.exporter.fmt=obj-mtl",
            "system.exporter.save_name=mesh",
        ]
    )
    return command


def find_latest_export_dir(trial_dir: str) -> str:
    save_dir = os.path.join(trial_dir, "save")
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"Save directory not found: {save_dir}")

    export_dirs = []
    for name in os.listdir(save_dir):
        if not name.endswith("-export"):
            continue
        match = re.match(r"it(\d+)-export$", name)
        if not match:
            continue
        export_dirs.append((int(match.group(1)), os.path.join(save_dir, name)))

    if not export_dirs:
        raise FileNotFoundError(f"No export directory found under {save_dir}")
    export_dirs.sort(key=lambda item: item[0])
    return export_dirs[-1][1]


def canonicalize_exported_assets(export_dir: str, sample_output_dir: str) -> None:
    os.makedirs(sample_output_dir, exist_ok=True)

    mesh_src = find_first_existing_file(export_dir, ("mesh.obj", "model.obj"))
    if mesh_src is not None:
        mesh_dst = os.path.join(sample_output_dir, "mesh.obj")
        shutil.copy2(mesh_src, mesh_dst)
        strip_obj_material_refs(mesh_dst)

    texture_sources = {
        "albedo.png": ("texture_kd.png", "texture_kd.jpg", "texture_kd.jpeg"),
        "metallic.png": (
            "texture_metallic.png",
            "texture_metallic.jpg",
            "texture_metallic.jpeg",
        ),
        "roughness.png": (
            "texture_roughness.png",
            "texture_roughness.jpg",
            "texture_roughness.jpeg",
        ),
        "normal.png": (
            "texture_nrm.png",
            "texture_nrm.jpg",
            "texture_nrm.jpeg",
            "texture_bump.png",
            "texture_bump.jpg",
            "texture_bump.jpeg",
        ),
    }

    for dst_name, candidates in texture_sources.items():
        src_path = find_first_existing_file(export_dir, candidates)
        if src_path is None:
            continue
        dst_path = os.path.join(sample_output_dir, dst_name)
        convert_or_copy_texture(src_path, dst_path)

    prune_sample_output_dir(sample_output_dir)


def find_first_existing_file(base_dir: str, candidates: Tuple[str, ...]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            return path
    return None


def convert_or_copy_texture(src_path: str, dst_path: str) -> None:
    if os.path.splitext(src_path)[1].lower() == ".png":
        shutil.copy2(src_path, dst_path)
        return
    with Image.open(src_path) as image:
        image.save(dst_path, format="PNG")


def strip_obj_material_refs(obj_path: str) -> None:
    with open(obj_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(obj_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("mtllib ") or line.startswith("usemtl "):
                continue
            f.write(line)


def prune_sample_output_dir(sample_output_dir: str) -> None:
    keep_files = {"albedo.png", "metallic.png", "normal.png", "roughness.png", "mesh.obj"}
    for name in os.listdir(sample_output_dir):
        path = os.path.join(sample_output_dir, name)
        if name in keep_files:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def validate_sample_output(sample_output_dir: str) -> None:
    missing = [
        name
        for name in REQUIRED_OUTPUT_FILES
        if not os.path.isfile(os.path.join(sample_output_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required exported files in {sample_output_dir}: {', '.join(missing)}"
        )


def check_sample_completed(sample_dir: str, source_mesh_path: Optional[str] = None) -> bool:
    return all(
        os.path.isfile(os.path.join(sample_dir, name)) for name in REQUIRED_OUTPUT_FILES
    )


def resolve_fix_view_num_override(extras: List[str]) -> int:
    default_fix_view_num = 128
    for extra in reversed(extras):
        if not extra.startswith("data.fix_view_num="):
            continue
        _, value = extra.split("=", 1)
        try:
            return max(1, int(str(value).strip().strip('"').strip("'")))
        except ValueError:
            return default_fix_view_num
    return default_fix_view_num


def check_prerender_completed(trial_dir: str, fix_view_num: int) -> bool:
    pre_render_dir = os.path.join(trial_dir, "pre_render")
    depth_dir = os.path.join(pre_render_dir, "depth")
    if not os.path.isdir(depth_dir):
        return False

    expected_last = os.path.join(depth_dir, f"{fix_view_num - 1:03d}.png")
    if os.path.isfile(expected_last):
        return True

    png_count = sum(
        1 for name in os.listdir(depth_dir) if name.lower().endswith(".png")
    )
    return png_count >= fix_view_num


def build_result_row(
    obj_id: str,
    sample_dir: str,
    caption_short: Optional[str] = None,
    caption_long: Optional[str] = None,
    caption_used: Optional[str] = None,
) -> Dict[str, str]:
    sample_dir = os.path.abspath(sample_dir)

    def path_if_exists(name: str) -> str:
        path = os.path.join(sample_dir, name)
        return os.path.abspath(path) if os.path.exists(path) else ""

    row = {
        "obj_id": obj_id,
        "mesh": path_if_exists("mesh.obj"),
        "albedo": path_if_exists("albedo.png"),
        "rough": path_if_exists("roughness.png"),
        "metal": path_if_exists("metallic.png"),
        "normal": path_if_exists("normal.png"),
    }
    if caption_short is not None:
        row["caption_short"] = caption_short
    if caption_long is not None:
        row["caption_long"] = caption_long
    if caption_used is not None:
        row["caption_used"] = caption_used
    return row


def append_to_manifest(tsv_path: str, new_rows: List[Dict[str, str]]) -> None:
    if not new_rows:
        return

    existing_ids = set()
    existing_rows: List[Dict[str, str]] = []
    if os.path.isfile(tsv_path):
        with open(tsv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                existing_ids.add(row.get("obj_id", ""))
                existing_rows.append(row)

    rows_to_add = [row for row in new_rows if row["obj_id"] not in existing_ids]
    if not rows_to_add:
        return

    all_rows = existing_rows + rows_to_add
    fieldnames = ["obj_id", "mesh", "albedo", "rough", "metal", "normal"]
    for name in ("caption_short", "caption_long", "caption_used"):
        if any(name in row for row in all_rows):
            fieldnames.append(name)

    os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return obj.item()
    except Exception:
        return str(obj)


def build_options_payload(
    args,
    filtered_extras: List[str],
    batch_root: str,
    tsv_path: str,
    result_tsv_path: str,
    effective_gpu_ids: List[int],
    workers_per_gpu: int,
    total_workers: int,
) -> Dict[str, Any]:
    return {
        "config": os.path.abspath(args.config),
        "gpu": args.gpu,
        "gpu_ids": args.gpu_ids,
        "effective_gpu_ids": effective_gpu_ids,
        "num_gpus": len(effective_gpu_ids),
        "workers_per_gpu": workers_per_gpu,
        "total_workers": total_workers,
        "train": bool(args.train),
        "validate": bool(args.validate),
        "test": bool(args.test),
        "export": bool(args.export),
        "gradio": True,
        "caption_field": args.caption_field,
        "tsv_path": os.path.abspath(tsv_path),
        "result_tsv": os.path.abspath(result_tsv_path),
        "max_samples": args.max_samples,
        "keep_intermediate": args.keep_intermediate,
        "batch_work_dir_name": args.batch_work_dir_name,
        "batch_textures_dir_name": args.batch_textures_dir_name,
        "batch_root": os.path.abspath(batch_root),
        "forwarded_overrides": list(filtered_extras),
    }


def save_batch_config(
    exp_dir: str,
    options: Dict[str, Any],
    processed_samples: List[Dict[str, Any]],
    skipped_samples: Optional[List[Dict[str, Any]]] = None,
    manifest_path: Optional[str] = None,
    timing_info: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "options": to_jsonable(options),
        "config": options.get("config"),
        "tsv_path": options.get("tsv_path"),
        "save_image": False,
        "processed_samples": processed_samples,
    }
    if skipped_samples:
        payload["skipped_samples"] = skipped_samples
    if manifest_path:
        payload["result_tsv"] = os.path.abspath(manifest_path)
    if timing_info:
        payload["timing"] = timing_info

    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def merge_and_save_batch_config(
    exp_dir: str,
    options: Dict[str, Any],
    new_processed: List[Dict[str, Any]],
    new_skipped: Optional[List[Dict[str, Any]]] = None,
    manifest_path: Optional[str] = None,
    timing_info: Optional[Dict[str, Any]] = None,
) -> None:
    config_path = os.path.join(exp_dir, "config.json")
    existing: Optional[Dict[str, Any]] = None
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    if existing is None:
        save_batch_config(
            exp_dir=exp_dir,
            options=options,
            processed_samples=new_processed,
            skipped_samples=new_skipped,
            manifest_path=manifest_path,
            timing_info=timing_info,
        )
        return

    prev_processed = existing.get("processed_samples", [])
    processed_ids = {sample.get("obj_id", "") for sample in prev_processed}
    for sample in new_processed:
        if sample.get("obj_id", "") not in processed_ids:
            prev_processed.append(sample)
            processed_ids.add(sample.get("obj_id", ""))
    prev_processed.sort(key=lambda item: item.get("obj_id", ""))

    prev_skipped = existing.get("skipped_samples", [])
    newly_processed_ids = {sample.get("obj_id", "") for sample in new_processed}
    merged_skipped = [
        sample
        for sample in prev_skipped
        if sample.get("obj_id", "") not in newly_processed_ids
    ]
    skipped_ids = {sample.get("obj_id", "") for sample in merged_skipped}
    for sample in new_skipped or []:
        if sample.get("obj_id", "") not in skipped_ids:
            merged_skipped.append(sample)
            skipped_ids.add(sample.get("obj_id", ""))
    merged_skipped.sort(key=lambda item: item.get("obj_id", ""))

    timing_key = "timing"
    suffix = 2
    while timing_key in existing:
        timing_key = f"timing{suffix}"
        suffix += 1

    existing["options"] = to_jsonable(options)
    existing["config"] = options.get("config")
    existing["tsv_path"] = options.get("tsv_path")
    existing["save_image"] = False
    existing["processed_samples"] = prev_processed
    existing["skipped_samples"] = merged_skipped
    if manifest_path:
        existing["result_tsv"] = os.path.abspath(manifest_path)
    if timing_info:
        existing[timing_key] = timing_info

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def cleanup_empty_dir(path: str) -> None:
    if not os.path.isdir(path):
        return
    if os.listdir(path):
        return
    os.rmdir(path)
