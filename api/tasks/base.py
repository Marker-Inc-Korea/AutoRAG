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

        # SQLite DB 업데이트
        project_db = SQLiteProjectDB(project_id)
        trial = project_db.get_trial(trial_id)
        if trial:
            if task_type == "evaluate" and status == Status.COMPLETED:
                trial.status = Status.COMPLETED
            project_db.set_trial(trial)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        super().on_failure(exc, task_id, args, kwargs, einfo)
        # Add your error handling logic here

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        super().on_success(retval, task_id, args, kwargs)
        # Add your success handling logic here
