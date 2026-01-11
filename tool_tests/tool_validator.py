from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from voice_agent_v4.config import SETTINGS
from voice_agent_v4.mcp_tools_server import get_mcp_server_async
from voice_agent_v4.providers.openai_adapter import _convert_tools_for_responses

from .logger import configure_logger

logger = configure_logger()


@dataclass
class SchemaIssue:
    tool: str
    path: str
    message: str


@dataclass
class ToolTestResult:
    tool: str
    success: bool
    mode: str
    message: str
    request_id: Optional[str] = None
    raw_error: Optional[str] = None
    schema_issues: List[SchemaIssue] = field(default_factory=list)


class ToolValidator:
    """
    Validates MCP tool schemas and performs lightweight OpenAI Responses API smoke tests.
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=SETTINGS.openai_api_key,
            base_url=SETTINGS.openai_base_url,
            organization=SETTINGS.openai_organization or None,
            timeout=SETTINGS.llm_stream_timeout_s,
        )

    async def load_tools(self) -> List[Dict[str, Any]]:
        mcp_server = await get_mcp_server_async()
        return mcp_server.list_tools()

    async def load_converted_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted = _convert_tools_for_responses(
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
                for tool in tools
            ]
        )
        logger.debug("Converted %d tools for Responses API", len(converted))
        return converted

    def run_schema_checks(self, converted_tools: List[Dict[str, Any]]) -> List[SchemaIssue]:
        issues: List[SchemaIssue] = []
        for entry in converted_tools:
            fn = entry.get("function", {})
            name = fn.get("name", "<unknown>")
            params = fn.get("parameters", {})
            issues.extend(self._validate_parameters_schema(name, params))
        return issues

    def _validate_parameters_schema(self, tool: str, schema: Dict[str, Any]) -> List[SchemaIssue]:
        issues: List[SchemaIssue] = []

        if schema.get("type") != "object":
            issues.append(SchemaIssue(tool, "$", "Root schema type must be 'object'."))

        properties = schema.get("properties")
        if properties is None:
            properties = {}
            schema["properties"] = properties

        if not isinstance(properties, dict):
            issues.append(SchemaIssue(tool, "$", "'properties' must be an object."))
            return issues

        if not properties:
            if schema.get("required"):
                issues.append(
                    SchemaIssue(tool, "$", "Schema with no properties should not declare 'required'.")
                )
            return issues

        required = schema.get("required", [])
        if not isinstance(required, list):
            issues.append(SchemaIssue(tool, "$", "'required' must be a list."))
            required = []

        for key, prop_schema in properties.items():
            path = f"$.properties.{key}"
            if not isinstance(prop_schema, dict):
                issues.append(SchemaIssue(tool, path, "Property schema must be an object."))
                continue

            prop_type = prop_schema.get("type")
            enum = prop_schema.get("enum")

            if enum and prop_type != "string":
                issues.append(SchemaIssue(tool, path, "Enum properties should have type 'string'."))

            if key not in required and not self._is_nullable(prop_schema):
                issues.append(
                    SchemaIssue(
                        tool,
                        path,
                        "Property not marked as required and not explicitly nullable; "
                        "Responses API may treat it as required. Consider adding 'required'.",
                    )
                )

        if schema.get("additionalProperties") is not False:
            issues.append(
                SchemaIssue(tool, "$", "'additionalProperties' must be false to satisfy strict tools.")
            )

        return issues

    @staticmethod
    def _is_nullable(prop_schema: Dict[str, Any]) -> bool:
        any_of = prop_schema.get("anyOf")
        if isinstance(any_of, list):
            for candidate in any_of:
                if isinstance(candidate, dict) and candidate.get("type") == "null":
                    return True
        return False

    async def run_remote_test(
        self, tools_payload: List[Dict[str, Any]], *, label: str
    ) -> ToolTestResult:
        input_payload = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are a calibration assistant. Do not execute tools; confirm readiness.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Acknowledge receipt of tool definitions.",
                    }
                ],
            },
        ]

        try:
            async with self.client.responses.stream(
                model=SETTINGS.openai_model,
                input=input_payload,
                tools=tools_payload,
                max_output_tokens=32,
                store=False,
            ) as stream:
                request_id: Optional[str] = None
                async for event in stream:
                    if not request_id:
                        request_id = getattr(event, "id", None)
                        if not request_id:
                            response_attr = getattr(event, "response", None)
                            if isinstance(response_attr, dict):
                                request_id = response_attr.get("id")
                            else:
                                request_id = getattr(response_attr, "id", None)
                    if getattr(event, "type", None) == "response.completed":
                        return ToolTestResult(
                            tool=label,
                            success=True,
                            mode="responses",
                            message="Accepted by Responses API",
                            request_id=request_id,
                        )
                return ToolTestResult(
                    tool=label,
                    success=True,
                    mode="responses",
                    message="Stream finished without errors",
                    request_id=request_id,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Responses API failure for %s: %s", label, exc, exc_info=True)
            return ToolTestResult(
                tool=label,
                success=False,
                mode="responses",
                message="Responses API rejected the request",
                raw_error=str(exc),
            )

    async def validate_all_tools(self) -> Tuple[List[ToolTestResult], List[SchemaIssue]]:
        raw_tools = await self.load_tools()
        converted_tools = await self.load_converted_tools(raw_tools)

        schema_issues = self.run_schema_checks(converted_tools)
        aggregated_schema: Dict[str, List[SchemaIssue]] = {}
        for issue in schema_issues:
            aggregated_schema.setdefault(issue.tool, []).append(issue)

        results: List[ToolTestResult] = []

        # First, test the entire catalog
        full_result = await self.run_remote_test(converted_tools, label="__FULL_CATALOG__")
        full_result.schema_issues = aggregated_schema.get("__FULL_CATALOG__", [])
        results.append(full_result)

        # Then, test each tool individually to pinpoint failures
        for tool_entry, converted_entry in zip(raw_tools, converted_tools, strict=False):
            name = tool_entry["name"]
            single_result = await self.run_remote_test([converted_entry], label=name)
            single_result.schema_issues = aggregated_schema.get(name, [])
            results.append(single_result)

        return results, schema_issues

    def export_results(self, results: List[ToolTestResult], schema_issues: List[SchemaIssue]) -> str:
        payload = {
            "results": [
                {
                    "tool": result.tool,
                    "success": result.success,
                    "mode": result.mode,
                    "message": result.message,
                    "request_id": result.request_id,
                    "raw_error": result.raw_error,
                    "schema_issues": [
                        {"path": issue.path, "message": issue.message} for issue in result.schema_issues
                    ],
                }
                for result in results
            ],
            "schema_issues": [
                {"tool": issue.tool, "path": issue.path, "message": issue.message}
                for issue in schema_issues
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)


async def run_validation() -> None:
    validator = ToolValidator()
    results, schema_issues = await validator.validate_all_tools()

    for result in results:
        status = "PASS" if result.success else "FAIL"
        logger.info("[%s] %s — %s", status, result.tool, result.message)
        for issue in result.schema_issues:
            logger.warning("  Schema issue %s: %s", issue.path, issue.message)
        if result.raw_error:
            logger.error("  Raw error: %s", result.raw_error)

    summary = validator.export_results(results, schema_issues)
    logger.info("Validation summary:\n%s", summary)


if __name__ == "__main__":
    asyncio.run(run_validation())
