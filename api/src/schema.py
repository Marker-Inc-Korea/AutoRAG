from datetime import datetime
from enum import Enum
from typing import Dict, Literal, Any, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


class TrialCreateRequest(BaseModel):
    name: Optional[str] = Field(None, description="The name of the trial")
    raw_path: Optional[str] = Field(None, description="The path to the raw data")
    corpus_path: Optional[str] = Field(None, description="The path to the corpus data")
    qa_path: Optional[str] = Field(None, description="The path to the QA data")
    config: Optional[Dict] = Field(None, description="The trial configuration dictionary")


class ParseRequest(BaseModel):
    config: Dict = Field(...,
                                                      description="Dictionary contains parse YAML configuration")
    name: str = Field(..., description="Name of the parse target dataset")
    path: str  # 추가: 파싱할 파일 경로

class ChunkRequest(BaseModel):
    config: Dict = Field(..., description="Dictionary contains chunk YAML configuration")
    name: str = Field(..., description="Name of the chunk target dataset")


class QACreationPresetEnum(str, Enum):
    BASIC = "basic"
    SIMPLE = "simple"
    ADVANCED = "advanced"

class LLMConfig(BaseModel):
    llm_name: str = Field(description="Name of the LLM model")
    llm_params: dict = Field(description="Parameters for the LLM model",
                             default={})

class SupportLanguageEnum(str, Enum):
    ENGLISH = "en"
    KOREAN = "ko"
    JAPANESE = "ja"

class QACreationRequest(BaseModel):
    preset: QACreationPresetEnum
    name: str = Field(..., description="Name of the QA dataset")
    qa_num: int
    llm_config: LLMConfig = Field(
        description="LLM configuration settings"
    )
    lang: SupportLanguageEnum = Field(default=SupportLanguageEnum.ENGLISH, description="Language of the QA dataset")

class EnvVariableRequest(BaseModel):
    key: str
    value: str

class Project(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    status: Literal["active", "archived"]
    metadata: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "id": "proj_123",
                "name": "My Project",
                "description": "A sample project",
                "created_at": "2024-02-11T12:00:00Z",
                "status": "active",
                "metadata": {}
            }
        }

class Status(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"

class TaskType(str, Enum):
    PARSE = "parse"
    CHUNK = "chunk"
    QA = "qa"
    VALIDATE = "validate"
    EVALUATE = "evaluate"
    REPORT = "report"
    CHAT = "chat"

class Task(BaseModel):
    id: str = Field(description="The task id")
    project_id: str
    trial_id: str = Field(description="The trial id", default="")
    name: Optional[str] = Field(None, description="The name of the task")
    config_yaml: Optional[Dict] = Field(
        None,
        description="YAML configuration. Format is dictionary, not path of the YAML file."
    )
    status: Status
    error_message: Optional[str] = Field(None, description="Error message if the task failed")
    type: TaskType
    created_at: Optional[datetime] = None
    save_path: Optional[str] = Field(
        None,
        description="Path where the task results are saved. It will be directory or file."
    )

class TrialConfig(BaseModel):
    trial_id: str
    project_id: str
    raw_path: Optional[str]
    corpus_path: Optional[str]
    qa_path: Optional[str]
    config_path: Optional[str]
    metadata: Dict = {}  # Using Dict as the default empty dict for metadata

    class Config:
        arbitrary_types_allowed = True

class Trial(BaseModel):
    id: str
    project_id: str
    config: Optional[TrialConfig] = Field(description="The trial configuration",
                                          default=None)
    name: str
    status: Status
    created_at: datetime
    report_task_id: Optional[str] = Field(None, description="The report task id for forcing shutdown of the task")
    chat_task_id: Optional[str] = Field(None, description="The chat task id for forcing shutdown of the task")

    corpus_path: Optional[str] = None
    qa_path: Optional[str] = None
    
    @field_validator('report_task_id', 'chat_task_id', mode="before")
    def replace_nan_with_none(cls, v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v
