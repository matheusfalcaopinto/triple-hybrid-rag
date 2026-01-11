"""Pydantic schemas package."""

from control_plane.schemas.agent import (
    AgentCreate,
    AgentDeploymentResponse,
    AgentResponse,
    AgentUpdate,
    AgentVersionCreate,
    AgentVersionPublish,
    AgentVersionResponse,
)
from control_plane.schemas.auth import (
    ForgotPasswordRequest,
    LoginRequest,
    LoginResponse,
    MeResponse,
    ResetPasswordRequest,
    UserInviteRequest,
    UserInviteResponse,
    UserMembership,
    UserResponse,
    UserUpdate,
)
from control_plane.schemas.call import (
    ActiveCallsResponse,
    CallEventResponse,
    CallHandoffRequest,
    CallHandoffResponse,
    CallResponse,
    CallsListResponse,
    OutboundCallRequest,
    TranscriptResponse,
    TranscriptSegmentResponse,
)
from control_plane.schemas.campaign import (
    CampaignCreate,
    CampaignResponse,
    CampaignUpdate,
)
from control_plane.schemas.common import (
    ErrorDetail,
    ErrorResponse,
    PaginatedResponse,
    SuccessResponse,
)
from control_plane.schemas.establishment import (
    EstablishmentCreate,
    EstablishmentResponse,
    EstablishmentUpdate,
    TelephonyPolicyResponse,
    TelephonyPolicyUpdate,
)
from control_plane.schemas.ingest import (
    CallEventPayload,
    IngestCallEventRequest,
    IngestResponse,
)
from control_plane.schemas.integration import (
    IntegrationConnectRequest,
    IntegrationConnectResponse,
    IntegrationConnectionResponse,
    IntegrationResponse,
)
from control_plane.schemas.lead import (
    LeadCreate,
    LeadImportResponse,
    LeadResponse,
    LeadUpdate,
)
from control_plane.schemas.report import (
    AgentPerformanceReport,
    DashboardKPIs,
)
from control_plane.schemas.runtime import (
    RuntimeCreateRequest,
    RuntimeResponse,
    RuntimeStatusResponse,
)

__all__ = [
    # Common
    "ErrorDetail",
    "ErrorResponse",
    "PaginatedResponse",
    "SuccessResponse",
    # Auth
    "LoginRequest",
    "LoginResponse",
    "ForgotPasswordRequest",
    "ResetPasswordRequest",
    "MeResponse",
    "UserResponse",
    "UserMembership",
    "UserInviteRequest",
    "UserInviteResponse",
    "UserUpdate",
    # Establishment
    "EstablishmentCreate",
    "EstablishmentResponse",
    "EstablishmentUpdate",
    "TelephonyPolicyResponse",
    "TelephonyPolicyUpdate",
    # Agent
    "AgentCreate",
    "AgentResponse",
    "AgentUpdate",
    "AgentVersionCreate",
    "AgentVersionResponse",
    "AgentVersionPublish",
    "AgentDeploymentResponse",
    # Runtime
    "RuntimeCreateRequest",
    "RuntimeResponse",
    "RuntimeStatusResponse",
    # Call
    "OutboundCallRequest",
    "CallResponse",
    "CallsListResponse",
    "ActiveCallsResponse",
    "CallEventResponse",
    "TranscriptSegmentResponse",
    "TranscriptResponse",
    "CallHandoffRequest",
    "CallHandoffResponse",
    # Lead
    "LeadCreate",
    "LeadResponse",
    "LeadUpdate",
    "LeadImportResponse",
    # Campaign
    "CampaignCreate",
    "CampaignResponse",
    "CampaignUpdate",
    # Report
    "DashboardKPIs",
    "AgentPerformanceReport",
    # Integration
    "IntegrationResponse",
    "IntegrationConnectRequest",
    "IntegrationConnectResponse",
    "IntegrationConnectionResponse",
    # Ingest
    "IngestCallEventRequest",
    "IngestResponse",
    "CallEventPayload",
]
