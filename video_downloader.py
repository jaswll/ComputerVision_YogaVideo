import json
import os
import time
import random
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# Set this to youtube-dl if you want to use youtube-dl.
youtube_downloader = "yt-dlp"

# Tuning knobs ----------------------------------------------------------------
# 4-8 workers is a safe range for YouTube. Higher risks rate-limits / soft IP
# throttling, which will actually slow you down overall.
MAX_WORKERS = 6

# Per-download hard cap (seconds). Prevents one stuck connection from
# blocking a worker forever. Lowest-resolution videos are small, so 10 min
# is generous.
DOWNLOAD_TIMEOUT = 600

# yt-dlp return codes / stderr patterns that mean "skip, don't retry".
# These are things like "Video unavailable", "Private video", "removed by
# uploader", geo-blocks, age-gates without login, etc. yt-dlp exits non-zero
# on all of them; we just log and move on.
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("download.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _already_downloaded(saveto, video_id):
    """yt-dlp picks the container based on the format, so check the common ones."""
    for ext in (".mp4", ".mkv", ".webm", ".flv", ".3gp"):
        if os.path.exists(os.path.join(saveto, video_id + ext)):
            return True
    return False


def _download_one(video_url, saveto, stats, lock, total):
    """Download a single video at the lowest available resolution.

    Returns a tuple: (status, video_url, message)
      status in {"ok", "skipped", "failed"}
    """
    video_id = video_url[-11:]

    if _already_downloaded(saveto, video_id):
        with lock:
            stats["skipped"] += 1
            done = stats["downloaded"] + stats["skipped"] + stats["failed"]
        logger.info(f"[{done/total*100:5.2f}%] SKIP {video_id} (exists)")
        return ("skipped", video_url, "already exists")

    # -f worst        : single lowest-quality combined stream (no ffmpeg merge needed)
    # --no-playlist   : guard against a URL that happens to include a list param
    # --no-warnings   : quieter logs
    # --socket-timeout: bail on hung sockets instead of stalling a worker
    # --retries       : yt-dlp's internal retry for transient HTTP errors
    # --no-overwrites : belt-and-suspenders against duplicate work across runs
    cmd = [
        youtube_downloader,
        video_url,
        "-f", "worst",
        "-o", os.path.join(saveto, "%(id)s.%(ext)s"),
        "--no-playlist",
        "--no-warnings",
        "--no-overwrites",
        "--socket-timeout", "30",
        "--retries", "3",
        "--fragment-retries", "3",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        with lock:
            stats["failed"] += 1
        logger.warning(f"TIMEOUT {video_id}")
        return ("failed", video_url, "timeout")
    except Exception as e:
        with lock:
            stats["failed"] += 1
        logger.error(f"ERROR {video_id}: {e}")
        return ("failed", video_url, str(e))

    if result.returncode == 0:
        with lock:
            stats["downloaded"] += 1
            done = stats["downloaded"] + stats["skipped"] + stats["failed"]
        logger.info(f"[{done/total*100:5.2f}%] OK   {video_id}")
        # Be nice to the host: small jittered pause per worker.
        time.sleep(random.uniform(1.0, 1.5))
        return ("ok", video_url, "")

    # Non-zero exit: private / deleted / unavailable / geo-blocked / etc.
    err_lines = (result.stderr or result.stdout or "").strip().splitlines()
    err_msg = err_lines[-1] if err_lines else f"exit {result.returncode}"
    with lock:
        stats["failed"] += 1
        done = stats["downloaded"] + stats["skipped"] + stats["failed"]
    logger.warning(f"[{done/total*100:5.2f}%] FAIL {video_id}: {err_msg}")
    return ("failed", video_url, err_msg)


def download_yt_videos(indexfile, saveto="raw_videos", max_workers=MAX_WORKERS):
    with open(indexfile) as f:
        content = json.load(f)

    os.makedirs(saveto, exist_ok=True)

    # Collect unique YouTube URLs up front. The source JSON can have the same
    # URL under multiple pose entries; no point queuing duplicate work.
    urls, seen = [], set()
    for entry in content:
        for inst in entry["instances"]:
            url = inst["url"]
            if "youtube" not in url and "youtu.be" not in url:
                continue
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)

    total = len(urls)
    logger.info(f"Found {total} unique YouTube URLs. Using {max_workers} workers.")

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    lock = Lock()
    failures = []

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_download_one, url, saveto, stats, lock, total): url
                for url in urls
            }
            try:
                for fut in as_completed(futures):
                    status, url, msg = fut.result()
                    if status == "failed":
                        failures.append((url, msg))
            except KeyboardInterrupt:
                # Default ThreadPoolExecutor.__exit__ waits for ALL queued
                # tasks to finish. With 5k queued, Ctrl+C would feel ignored.
                # cancel_futures=True drops the not-yet-started ones.
                logger.warning("Interrupted — cancelling queued downloads...")
                ex.shutdown(wait=False, cancel_futures=True)
                raise
    except KeyboardInterrupt:
        pass
    finally:
        # Persist the failure list so you can retry just the bad ones later.
        if failures:
            with open("failed_downloads.txt", "w", encoding="utf-8") as f:
                for url, msg in failures:
                    f.write(f"{url}\t{msg}\n")
            logger.info(f"Wrote {len(failures)} failures to failed_downloads.txt")

        logger.info(
            "Done. downloaded=%d  skipped=%d  failed=%d"
            % (stats["downloaded"], stats["skipped"], stats["failed"])
        )


if __name__ == "__main__":
    download_yt_videos("3DYoga90.json")
