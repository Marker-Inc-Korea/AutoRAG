from celery import shared_task
from .base import TrialTask

@shared_task(bind=True, base=TrialTask)
def parse_documents(self, project_id: str, trial_id: str):
    """문서 파싱 작업"""
    try:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status="parsing",
            progress=0,
            task_type="parse"
        )
        
        # 파싱 작업 수행
        # ...
        
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status="parsed",
            progress=100,
            task_type="parse"
        )
        
        return {"status": "success", "task_type": "parse"}
    except Exception as e:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status="parse_failed",
            progress=0,
            task_type="parse",
            info={"error": str(e)}
        )
        raise

@shared_task(bind=True, base=TrialTask)
def chunk_documents(self, project_id: str, trial_id: str):
    """문서 청킹 작업"""
    try:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status="chunking",
            progress=0,
            task_type="chunk"
        )
        
        # 청킹 작업 수행
        # ...
        
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status="chunked",
            progress=100,
            task_type="chunk"
        )
        
        return {"status": "success", "task_type": "chunk"}
    except Exception as e:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status="chunk_failed",
            progress=0,
            task_type="chunk",
            info={"error": str(e)}
        )
        raise