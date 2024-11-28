from celery import Task
from typing import Dict, Any
from database.project_db import SQLiteProjectDB
from src.schema import Status


class TrialTask(Task):
    """Trial 관련 모든 task의 기본 클래스"""

    def update_state_and_db(
        self,
        trial_id: str,
        project_id: str,
        status: str,
        progress: int,
        task_type: str,
        parse_task_id: str = None,
        chunk_task_id: str = None,
        info: Dict[str, Any] = None,
    ):
        """Task 상태와 DB를 함께 업데이트"""
        # Redis에 상태 업데이트 (Celery)
        self.update_state(
            state="PROGRESS",
            meta={
                "trial_id": trial_id,
                "task_type": task_type,
                "current": progress,
                "parse_task_id": parse_task_id,
                "chunk_task_id": chunk_task_id,
                "total": 100,
                "status": status,
                "info": info or {},
            },
        )

        # 상태 매핑 추가
        status_map = {
            "PENDING": Status.IN_PROGRESS,
            "STARTED": Status.IN_PROGRESS,
            "SUCCESS": Status.COMPLETED,
            "FAILURE": Status.FAILED,
            "chunking": Status.IN_PROGRESS,
            "parsing": Status.IN_PROGRESS,
            "generating_qa_docs": Status.IN_PROGRESS,
        }
        trial_status = status_map.get(status, Status.FAILED)

        # SQLite DB 업데이트
        project_db = SQLiteProjectDB(project_id)
        trial = project_db.get_trial(trial_id)
        if trial:
            trial.status = trial_status  # 매핑된 상태 사용
            if task_type == "parse":
                trial.parse_task_id = self.request.id
            elif task_type == "chunk":
                trial.chunk_task_id = self.request.id
            project_db.set_trial(trial)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        super().on_failure(exc, task_id, args, kwargs, einfo)
        # Add your error handling logic here

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        super().on_success(retval, task_id, args, kwargs)
        # Add your success handling logic here
