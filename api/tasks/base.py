from celery import Task
from typing import Dict, Any
from src.trial_config import SQLiteTrialDB
import os

class TrialTask(Task):
    """Trial 관련 모든 task의 기본 클래스"""
    
    def update_state_and_db(self, trial_id: str, project_id: str, 
                           status: str, progress: int, 
                           task_type: str, info: Dict[str, Any] = None):
        """Task 상태와 DB를 함께 업데이트"""
        # Redis에 상태 업데이트 (Celery)
        self.update_state(
            state='PROGRESS',
            meta={
                'trial_id': trial_id,
                'task_type': task_type,
                'current': progress,
                'total': 100,
                'status': status,
                'info': info or {}
            }
        )
        
        # SQLite DB 업데이트
        db_path = os.path.join("projects", project_id, "trials.db")
        trial_db = SQLiteTrialDB(db_path)
        trial = trial_db.get_trial(trial_id)
        if trial:
            trial.status = status
            if task_type == 'parse':
                trial.parse_task_id = self.request.id
            elif task_type == 'chunk':
                trial.chunk_task_id = self.request.id
            trial_db.set_trial(trial)
        
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        super().on_failure(exc, task_id, args, kwargs, einfo)
        # Add your error handling logic here
        
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        super().on_success(retval, task_id, args, kwargs)
        # Add your success handling logic here 