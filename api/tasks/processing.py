from .base import ProgressTask
from celery import shared_task
import time

@shared_task(bind=True, base=ProgressTask)
def process_documents(self, documents):
    """Example task with progress tracking"""
    total = len(documents)
    
    for i, doc in enumerate(documents, 1):
        # Process document
        time.sleep(1)  # Simulate work
        
        # Update progress
        self.update_progress(
            current=i,
            total=total,
            info={'current_doc': doc['id']}
        )
    
    return {'processed': total} 