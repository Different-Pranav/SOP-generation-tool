from pydantic import BaseModel
from typing import Optional, List


class StudentInfo(BaseModel):
    name: str
    background: str
    gpa: str
    work_experience: Optional[str] = None
    achievements: Optional[List[str]] = None
    background_story: str
    goals: str


class SOPRequest(BaseModel):
    student_info: StudentInfo
    university: str
    program: str


# Response Model
class SOPResponse(BaseModel):
    sop: str
