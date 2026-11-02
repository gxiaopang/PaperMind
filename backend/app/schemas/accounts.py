from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EmailInput(Input):
    email: EmailStr

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.casefold()


Password = Annotated[str, Field(min_length=8, max_length=128)]


class Credentials(EmailInput):
    password: Password

    @field_validator("password", mode="before")
    @classmethod
    def preserve_password(cls, value: str) -> str:
        # Password whitespace is significant, unlike ordinary form fields.
        return value

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class RegistrationInput(Credentials):
    code: str = Field(pattern=r"^[0-9]{6}$")


class EmailCodeInput(Input):
    code: str = Field(pattern=r"^[0-9]{6}$")


class TokenInput(Input):
    token: str = Field(min_length=32, max_length=256)


class ResetInput(TokenInput):
    password: Password
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class UserView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    email: str
    email_verified_at: datetime | None
    role: str
    status: str


class CollectionInput(Input):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)


class CollectionView(CollectionInput):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: datetime


class UserSettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_language: Literal["zh", "en"] = "zh"
    detail_level: Literal["concise", "standard", "detailed"] = "standard"
    memory_write_mode: Literal["auto", "review"] = "auto"


class AdminUserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["active", "disabled"]


class AdminQuotaUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_papers: int | None = Field(default=None, gt=0, le=10000)
    storage_mb: int | None = Field(default=None, gt=0, le=1048576)
    daily_parse_pages: int | None = Field(default=None, gt=0, le=100000)
    daily_model_tokens: int | None = Field(default=None, gt=0, le=100000000)
    upload_concurrency: int | None = Field(default=None, gt=0, le=50)
