"""

Violation Service - Phase 8

In-memory event repository for MalpracticeEvent.

Independent from FastAPI, designed for future DB replacement.

Thread-safe.

"""

import time

import json

import threading

import sqlite3

from contextlib import contextmanager

from dataclasses import fields

from pathlib import Path

from typing import Dict, List, Optional, Any

from collections import defaultdict



from app.services.rule_engine import MalpracticeEvent

from app.services.evidence_manager import _get_event_interval





class ViolationService:

    """

    Event registry. Stores MalpracticeEvent by event_id.

    Evidence lookup via filesystem (clips/evidence/<session>/<event_id>/).

    """



    def __init__(self, base_evidence_dir: str | Path | None = None):

        self._lock = threading.RLock()

        # event_id -> {"event": MalpracticeEvent, "session_id": str, "registered_at": float}

        self._events: Dict[str, dict] = {}

        # student_id -> set(event_id) for quick filter

        if base_evidence_dir is None:

            # Default to <project_root>/clips/evidence (resolve relative to this file)

            try:

                _root = Path(__file__).resolve().parents[3]

                base_evidence_dir = _root / "clips" / "evidence"

            except:

                base_evidence_dir = Path("clips/evidence")

        self.base_dir = Path(base_evidence_dir)

        project_root = Path(__file__).resolve().parents[3]

        from app.config.settings import settings

        configured_path = Path(settings.DATABASE_PATH)

        self.database_path = configured_path if configured_path.is_absolute() else project_root / configured_path

        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

        self._load_persisted_events()



    @contextmanager

    def _connect(self):

        connection = sqlite3.connect(self.database_path, timeout=30)

        connection.row_factory = sqlite3.Row

        try:

            yield connection

            connection.commit()

        except Exception:

            connection.rollback()

            raise

        finally:

            connection.close()



    def _initialize_database(self):

        with self._connect() as connection:

            connection.execute("PRAGMA journal_mode=WAL")

            connection.execute("""

                CREATE TABLE IF NOT EXISTS violations (

                    event_id TEXT PRIMARY KEY,

                    session_id TEXT NOT NULL,

                    event_json TEXT NOT NULL,

                    registered_at REAL NOT NULL

                )

            """)

            connection.execute("""

                CREATE TABLE IF NOT EXISTS evidence (

                    event_id TEXT PRIMARY KEY REFERENCES violations(event_id) ON DELETE CASCADE,

                    session_id TEXT NOT NULL,

                    metadata_json TEXT NOT NULL,

                    video BLOB,

                    created_at TEXT NOT NULL

                )

            """)

            connection.execute("""

                CREATE TABLE IF NOT EXISTS violation_reviews (

                    event_id TEXT PRIMARY KEY REFERENCES violations(event_id) ON DELETE CASCADE,

                    reviewer TEXT NOT NULL,

                    action TEXT NOT NULL CHECK(action IN ('CONFIRMED', 'REJECTED')),

                    notes TEXT NOT NULL DEFAULT '',

                    reviewed_at REAL NOT NULL

                )

            """)



    def save_review(self, event_id: str, action: str, reviewer: str, notes: str) -> Optional[dict]:

        if not self.get_event(event_id):

            return None

        reviewed_at = time.time()

        with self._connect() as connection:

            connection.execute(

                """INSERT INTO violation_reviews (event_id, reviewer, action, notes, reviewed_at)

                   VALUES (?, ?, ?, ?, ?) ON CONFLICT(event_id) DO UPDATE SET

                   reviewer=excluded.reviewer, action=excluded.action, notes=excluded.notes,

                   reviewed_at=excluded.reviewed_at""",

                (event_id, reviewer, action, notes, reviewed_at),

            )

        return {"event_id": event_id, "reviewer": reviewer, "action": action, "notes": notes, "reviewed_at": reviewed_at}



    def get_review(self, event_id: str) -> Optional[dict]:

        with self._connect() as connection:

            row = connection.execute("SELECT * FROM violation_reviews WHERE event_id = ?", (event_id,)).fetchone()

        return dict(row) if row else None



    def _load_persisted_events(self):

        from app.services.rule_engine import EventStatus, Severity



        with self._connect() as connection:

            rows = connection.execute("SELECT * FROM violations").fetchall()

        for row in rows:

            try:

                values = json.loads(row["event_json"])

                values["status"] = EventStatus(values["status"])

                values["severity"] = Severity(values["severity"])

                self._events[row["event_id"]] = {

                    "event": MalpracticeEvent(**values),

                    "session_id": row["session_id"],

                    "registered_at": row["registered_at"],

                }

            except (TypeError, ValueError, KeyError, json.JSONDecodeError):

                continue



    @staticmethod

    def _event_json(event):

        values = {field.name: getattr(event, field.name) for field in fields(event)}

        for key, value in values.items():

            if hasattr(value, "value"):

                values[key] = value.value

        return json.dumps(values, default=str)



    
    def store_evidence(self, event_id: str, session_id: str, metadata: dict, video_path=None):
        """Persist evidence without deleting an existing video on metadata-only updates."""
        video = None

        if video_path:
            path = Path(video_path)
            if path.is_file():
                try:
                    candidate = path.read_bytes()
                    if candidate:
                        video = candidate
                except OSError:
                    # Preserve any existing video if the replacement cannot be read.
                    video = None

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT video FROM evidence WHERE event_id = ?",
                (event_id,),
            ).fetchone()

            # A missing replacement must not erase a previously saved video.
            if video is None and existing is not None:
                video = existing["video"]

            connection.execute(
                """
                INSERT INTO evidence
                    (event_id, session_id, metadata_json, video, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    metadata_json = excluded.metadata_json,
                    video = COALESCE(excluded.video, evidence.video),
                    created_at = excluded.created_at
                """,
                (
                    event_id,
                    session_id,
                    json.dumps(metadata, default=str),
                    video,
                    time.time(),
                ),
            )




    def get_stored_evidence(self, event_id: str):

        with self._connect() as connection:

            row = connection.execute("SELECT * FROM evidence WHERE event_id = ?", (event_id,)).fetchone()

        if not row:

            return None

        return {

            "session_id": row["session_id"],

            "metadata": json.loads(row["metadata_json"]),

            "video": row["video"],

        }



    def clear(self):

        """For tests - clear all."""

        with self._lock:

            self._events.clear()

            with self._connect() as connection:

                connection.execute("DELETE FROM evidence")

                connection.execute("DELETE FROM violation_reviews")

                connection.execute("DELETE FROM violations")



    def register_event(self, event: MalpracticeEvent, session_id: str = "default_session") -> bool:

        """

        Register event. Returns False if duplicate (already exists).

        Uses event_id as unique key.

        """

        if event is None or not getattr(event, "event_id", None):

            raise ValueError("event must have event_id")

        eid = str(event.event_id)

        # Reject path traversal in event_id

        if "/" in eid or "\\" in eid or ".." in eid:

            raise ValueError("invalid event_id")

        with self._lock:

            if eid in self._events:

                return False

            # Safe session_id

            safe_session = str(session_id).replace("/", "_").replace("\\", "_")

            self._events[eid] = {

                "event": event,

                "session_id": safe_session,

                "registered_at": time.time(),

            }

            with self._connect() as connection:

                connection.execute(

                    "INSERT OR IGNORE INTO violations (event_id, session_id, event_json, registered_at) VALUES (?, ?, ?, ?)",

                    (eid, safe_session, self._event_json(event), time.time()),

                )

            return True



    def update_event(self, event: MalpracticeEvent, session_id: Optional[str] = None) -> bool:

        """Persist the latest lifecycle state of an already-registered event."""
        if event is None or not getattr(event, "event_id", None):
            return False
        eid = str(event.event_id)
        if not eid or "/" in eid or "\\" in eid or ".." in eid:
            return False

        with self._lock:
            existing = self._events.get(eid)
            if existing is None:
                return False
            safe_session = str(session_id if session_id is not None else existing.get("session_id", "default_session"))
            safe_session = safe_session.replace("/", "_").replace("\\", "_")
            self._events[eid] = {
                "event": event,
                "session_id": safe_session,
                "registered_at": existing.get("registered_at", time.time()),
            }
            with self._connect() as connection:
                connection.execute(
                    "UPDATE violations SET session_id = ?, event_json = ? WHERE event_id = ?",
                    (safe_session, self._event_json(event), eid),
                )
            return True

    def get_event(self, event_id: str) -> Optional[dict]:

        """Return dict with event, session_id or None."""

        if not event_id or "/" in event_id or "\\" in event_id or ".." in event_id:

            return None

        with self._lock:

            return self._events.get(str(event_id))



    def get_event_obj(self, event_id: str) -> Optional[MalpracticeEvent]:

        d = self.get_event(event_id)

        return d["event"] if d else None



    def get_session_id(self, event_id: str) -> Optional[str]:

        d = self.get_event(event_id)

        return d["session_id"] if d else None



    def list_events(

        self,

        student_id: Optional[int] = None,

        rule_id: Optional[str] = None,

        event_type: Optional[str] = None,

        status: Optional[str] = None,

        severity: Optional[str] = None,

        session_id: Optional[str] = None,

        start_time: Optional[float] = None,

        end_time: Optional[float] = None,

        limit: int = 50,

        offset: int = 0,

    ) -> List[dict]:

        """

        Filter and paginate. Returns list of dicts {event, session_id}.

        Filters are ANDed.

        Time filter uses event_start/end via _get_event_interval.

        """

        # Validate pagination

        if limit < 0 or limit > 1000:

            raise ValueError("limit must be 0..1000")

        if offset < 0:

            raise ValueError("offset must be >=0")

        with self._lock:

            items = list(self._events.values())

        # Apply filters

        filtered = []

        for item in items:

            ev = item["event"]

            sid = item["session_id"]

            if student_id is not None and getattr(ev, "student_id", None) != student_id:

                continue

            if rule_id is not None and getattr(ev, "rule_id", None) != rule_id:

                continue

            if event_type is not None and getattr(ev, "event_type", None) != event_type:

                continue

            if status is not None:

                ev_status = getattr(ev, "status", None)

                try:

                    ev_status = ev_status.value if hasattr(ev_status, "value") else str(ev_status)

                except:

                    ev_status = str(ev_status)

                if ev_status != status:

                    continue

            if severity is not None:

                ev_sev = getattr(ev, "severity", None)

                try:

                    ev_sev = ev_sev.value if hasattr(ev_sev, "value") else str(ev_sev)

                except:

                    ev_sev = str(ev_sev)

                if ev_sev != severity:

                    continue

            if session_id is not None and sid != session_id:

                continue

            # Time filtering: event interval overlaps [start_time, end_time] if provided

            if start_time is not None or end_time is not None:

                try:

                    es, ee = _get_event_interval(ev)

                except:

                    continue

                if start_time is not None and ee < float(start_time):

                    continue

                if end_time is not None and es > float(end_time):

                    continue

            filtered.append(item)

        # Sort by event_start descending (newest first) for consistent pagination

        def sort_key(item):

            try:

                es, _ = _get_event_interval(item["event"])

                return es

            except:

                return 0

        filtered.sort(key=sort_key, reverse=True)

        # Pagination

        return filtered[offset : offset + limit]



    def count_events(self, **filters) -> int:

        """Count matching filters without pagination limit."""

        filters.pop("limit", None)

        filters.pop("offset", None)

        # Directly filter without pagination limit to avoid 1000 cap

        # Reuse list_events logic but without limit/offset truncation

        # Call with large limit that bypasses validation by temporarily increasing allowed

        # Instead, implement counting directly:

        with self._lock:

            items = list(self._events.values())

        count = 0

        for item in items:

            ev = item["event"]

            sid = item["session_id"]

            if filters.get("student_id") is not None and getattr(ev, "student_id", None) != filters["student_id"]:

                continue

            if filters.get("rule_id") is not None and getattr(ev, "rule_id", None) != filters["rule_id"]:

                continue

            if filters.get("event_type") is not None and getattr(ev, "event_type", None) != filters["event_type"]:

                continue

            if filters.get("status") is not None:

                ev_status = getattr(ev, "status", None)

                try:

                    ev_status = ev_status.value if hasattr(ev_status, "value") else str(ev_status)

                except:

                    ev_status = str(ev_status)

                if ev_status != filters["status"]:

                    continue

            if filters.get("severity") is not None:

                ev_sev = getattr(ev, "severity", None)

                try:

                    ev_sev = ev_sev.value if hasattr(ev_sev, "value") else str(ev_sev)

                except:

                    ev_sev = str(ev_sev)

                if ev_sev != filters["severity"]:

                    continue

            if filters.get("session_id") is not None and sid != filters["session_id"]:

                continue

            st = filters.get("start_time")

            et = filters.get("end_time")

            if st is not None or et is not None:

                try:

                    es, ee = _get_event_interval(ev)

                except:

                    continue

                if st is not None and ee < float(st):

                    continue

                if et is not None and es > float(et):

                    continue

            count += 1

        return count



    
    def get_evidence_info(self, event_id: str) -> Optional[dict]:
        """
        Return separate video and metadata availability.
        Paths are derived from registered events and constrained to base_dir.
        """
        item = self.get_event(event_id)
        if not item:
            return None

        session_id = item["session_id"]
        stored = self.get_stored_evidence(event_id)

        safe_eid = str(event_id).replace("/", "_").replace("\\", "_")
        safe_sid = str(session_id).replace("/", "_").replace("\\", "_")

        base_resolved = self.base_dir.resolve()
        video_path = base_resolved / safe_sid / safe_eid / "original_evidence.mp4"
        meta_path = base_resolved / safe_sid / safe_eid / "metadata.json"

        def safe_file(path: Path) -> Optional[Path]:
            try:
                resolved = path.resolve()
                if not resolved.is_relative_to(base_resolved):
                    return None
                return resolved if resolved.is_file() else None
            except (OSError, ValueError, RuntimeError):
                return None

        resolved_video = safe_file(video_path)
        resolved_meta = safe_file(meta_path)

        # Support older evidence saved under a different session directory.
        if resolved_video is None and resolved_meta is None:
            try:
                if base_resolved.is_dir():
                    for session_dir in base_resolved.iterdir():
                        if not session_dir.is_dir():
                            continue

                        candidate_video = safe_file(
                            session_dir / safe_eid / "original_evidence.mp4"
                        )
                        candidate_meta = safe_file(
                            session_dir / safe_eid / "metadata.json"
                        )

                        if candidate_video is not None or candidate_meta is not None:
                            resolved_video = candidate_video
                            resolved_meta = candidate_meta
                            break
            except OSError:
                pass

        stored_video = bool(stored and stored.get("video"))
        stored_metadata = bool(stored and stored.get("metadata") is not None)

        video_exists = resolved_video is not None or stored_video
        metadata_exists = resolved_meta is not None or stored_metadata

        return {
            "event_id": event_id,
            "session_id": session_id,
            # Keep general evidence availability separate from playable video.
            "available": video_exists or metadata_exists,
            "video_exists": video_exists,
            "metadata_exists": metadata_exists,
            "video_path": str(resolved_video) if resolved_video else None,
            "metadata_path": str(resolved_meta) if resolved_meta else None,
        }




    def get_metadata(self, event_id: str) -> Optional[dict]:

        """Return parsed metadata.json dict or None."""

        info = self.get_evidence_info(event_id)

        stored = self.get_stored_evidence(event_id)

        if stored:

            return stored["metadata"]

        if not info or not info["metadata_exists"]:

            return None

        try:

            with open(info["metadata_path"], "r", encoding="utf-8") as f:

                return json.load(f)

        except:

            return None



    def get_summary(self) -> dict:

        """Aggregate summary for dashboard."""

        with self._lock:

            items = list(self._events.values())

        total = len(items)

        by_rule = defaultdict(int)

        by_severity = defaultdict(int)

        by_student = defaultdict(int)

        by_status = defaultdict(int)

        by_session = defaultdict(int)

        for item in items:

            ev = item["event"]

            by_rule[getattr(ev, "rule_id", "unknown")] += 1

            sev = getattr(ev, "severity", "unknown")

            try:

                sev = sev.value if hasattr(sev, "value") else str(sev)

            except:

                sev = str(sev)

            by_severity[sev] += 1

            by_student[str(getattr(ev, "student_id", "unknown"))] += 1

            st = getattr(ev, "status", "unknown")

            try:

                st = st.value if hasattr(st, "value") else str(st)

            except:

                st = str(st)

            by_status[st] += 1

            by_session[item["session_id"]] += 1

        return {

            "total_violations": total,

            "violations_by_rule": dict(by_rule),

            "violations_by_severity": dict(by_severity),

            "violations_by_student": dict(by_student),

            "violations_by_status": dict(by_status),

            "violations_by_session": dict(by_session),

        }



    def get_info(self) -> dict:

        with self._lock:

            return {"total_events": len(self._events), "base_dir": str(self.base_dir)}





# Global singleton for app

violation_service = ViolationService()
