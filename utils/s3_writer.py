from __future__ import annotations

"""
S3 writer queue: serializes heavy Parquet -> S3 writes in a background thread.

Benefits on RPi:
- Reduces visible bursts and stalls by moving CPU-heavy serialization off worker threads
- Ensures only one write is in-flight to S3 at a time (configurable by queue size)
"""

import io
import os
import threading
import queue
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class _WriteDfTask:
    key: str
    df: pd.DataFrame
    future: Future


class S3Writer:
    def __init__(self, s3_client, max_queue: int = 64):
        self.s3 = s3_client
        self.q: queue.Queue[_WriteDfTask | None] = queue.Queue(maxsize=max_queue)
        self._th = threading.Thread(target=self._run, name="S3Writer", daemon=True)
        self._started = False
        self._stopping = False
        # Prefer zstd for dense numerical data; fall back gracefully if unavailable
        self.compression = os.getenv("PARQUET_COMPRESSION", "zstd")

    def start(self):
        if not self._started:
            self._started = True
            self._th.start()

    def stop(self, timeout: Optional[float] = 5.0):
        if not self._started:
            return
        try:
            self._stopping = True
            self.q.put_nowait(None)
        except Exception:
            pass
        try:
            self._th.join(timeout=timeout)
        except Exception:
            pass

    def submit_df_parquet(self, key: str, df: pd.DataFrame) -> Future:
        fut: Future = Future()
        try:
            self.q.put(_WriteDfTask(key=key, df=df, future=fut))
        except Exception as e:
            fut.set_exception(e)
        return fut

    def _run(self):
        while True:
            try:
                task = self.q.get()
            except Exception:
                break
            if task is None:
                break
            try:
                self._handle_task(task)
                task.future.set_result(True)
            except Exception as e:
                logger.warning(f"S3Writer task failed for {task.key}: {e}")
                task.future.set_exception(e)
            finally:
                try:
                    self.q.task_done()
                except Exception:
                    pass

    def _handle_task(self, task: _WriteDfTask):
        key = task.key
        df = task.df
        # Convert to parquet bytes
        try:
            logger.debug(f"S3Writer: begin serialize parquet {key} rows={len(df)} compression={self.compression}")
        except Exception:
            logger.debug(f"S3Writer: begin serialize parquet {key}")
        buffer = io.BytesIO()
        try:
            df.to_parquet(buffer, index=False, compression=self.compression)
        except Exception as e:
            # Fallback to default compression if env-provided codec is unavailable
            logger.warning(f"Parquet compression '{self.compression}' failed ({e}); falling back to default codec")
            df.to_parquet(buffer, index=False)
        data = buffer.getvalue()
        # Upload with temp key pattern, idempotent if final exists
        temp_key = f"{key}.tmp"
        logger.debug(f"S3Writer: begin put tmp {temp_key} ({len(data)} bytes)")
        self.s3.put_object(temp_key, data)
        if self.s3.object_exists(key):
            # Already present; clean up temp and exit
            logger.debug(f"S3Writer: final exists, begin delete tmp {temp_key}")
            self.s3.delete_object(temp_key)
            logger.debug(f"S3Writer: final exists, skipped write {key}")
            return
        # Promote temp to final (copy+delete simulated by second put)
        logger.debug(f"S3Writer: begin promote final {key}")
        self.s3.put_object(key, data)
        logger.debug(f"S3Writer: begin delete tmp {temp_key}")
        self.s3.delete_object(temp_key)
        logger.info(f"S3Writer: wrote {key} ({len(data)} bytes)")
