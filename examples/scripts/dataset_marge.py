# python - <<'PY'
from pathlib import Path

from medmo_loader import load_and_merge_datasets, export_manifest_jsonl_chunked


TARGET_DIR = Path("/vast/users/imran.razzak/Medmo_Dataset/Medmo_Dataset/ALL_VQAs")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "medtrinity_report", "iuxray_report", "mimiccxr_report", "rocov2_report",
    # "chexpert_plus_report", "vqa_med_2019", "pubmed_vision", "medpix_cliqa_report",
    # "nih_vqa", "quilt_llava_pretrain", "mimic_cxr_vqa", "roco_report",
    # "nih_bbox", "deeplesion_bbox", "grazpedwri_dx_bbox_resize", "bacteria_bbox_resize",
    # "ctc_bbox_resize", "deepcell_bbox_resize", "medsg_bbox", "omnimed_vqa", "mmmu_med", "slake", "vqa_rad",
    # "path_vqa", "pmc_vqa", "slake_bbox",
]

# MedTrinity must stay streaming so it does not blow up RAM.
PER_DS_KWARGS = {
    "medtrinity_report": {"resume_from_step": 0, "batch_size": 32, "streaming": True},
}

# How many rows per JSONL chunk; adjust down if disk space is tight.
CHUNK_SIZE = 200_000


def run():
    for split in ["train", "validation", "test"]:
        try:
            merged = load_and_merge_datasets(
                DATASETS,
                split=split,
                strategy="interleave",
                seed=42,
                per_dataset_kwargs=PER_DS_KWARGS,
            )
        except ValueError:
            print(f"[{split}] no datasets exposed this split, skipping.")
            continue

        written_paths = export_manifest_jsonl_chunked(
            merged,
            out_dir=str(TARGET_DIR),
            split=split,
            chunk_size=CHUNK_SIZE,
            prefix=split,
        )
        print(f"[{split}] wrote {len(written_paths)} chunk(s) to {TARGET_DIR}")


if __name__ == "__main__":
    run()
