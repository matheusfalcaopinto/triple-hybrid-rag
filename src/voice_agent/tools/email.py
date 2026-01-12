from __future__ import annotations

import asyncio
import json
import logging
import os
import smtplib
import uuid
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, Optional

from voice_agent.communication import events as comm_events
from voice_agent.communication.storage import CommunicationEvent, record_event_sync

logger = logging.getLogger("voice_agent_v4.email")


def get_smtp_config() -> Dict[str, Any]:
    """
    Get SMTP configuration from environment variables.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT", "587")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM_EMAIL") or smtp_user

    if not smtp_host:
        raise ValueError(
            "SMTP_HOST not configured. "
            "Set your SMTP server in .env (e.g., smtp.gmail.com)"
        )
    if not smtp_user:
        raise ValueError(
            "SMTP_USER not configured. "
            "Set your SMTP username/email in .env"
        )
    if not smtp_password:
        raise ValueError(
            "SMTP_PASSWORD not configured. "
            "Set your SMTP password/app password in .env"
        )

    return {
        "host": smtp_host,
        "port": int(smtp_port),
        "user": smtp_user,
        "password": smtp_password,
        "from_email": smtp_from,
    }


def _send_email_blocking(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    html: bool = False,
    *,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous email sender used in background threads.
    """
    try:
        config = get_smtp_config()

        msg = MIMEMultipart()
        msg["From"] = config["from_email"]
        msg["To"] = to
        msg["Subject"] = subject

        if cc:
            msg["Cc"] = cc
        if bcc:
            msg["Bcc"] = bcc

        mime_type = "html" if html else "plain"
        msg.attach(MIMEText(body, mime_type))

        recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
        if cc:
            recipients.extend([addr.strip() for addr in cc.split(",") if addr.strip()])
        if bcc:
            recipients.extend([addr.strip() for addr in bcc.split(",") if addr.strip()])

        if not recipients:
            raise ValueError("No recipients provided")

        timeout = float(os.getenv("SMTP_TIMEOUT", "15"))
        with smtplib.SMTP(config["host"], config["port"], timeout=timeout) as server:
            server.starttls()
            server.login(config["user"], config["password"])
            try:
                server.send_message(msg, to_addrs=recipients)
            except TypeError:
                server.send_message(msg)

        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="email",
                status="sent",
                payload={
                    "to": recipients,
                    "subject": subject,
                },
            )

        return {
            "success": True,
            "message": "Email sent successfully",
            "to": ",".join(recipients),
            "subject": subject,
            "correlation_id": correlation_id,
            "channel": "email",
        }

    except (ValueError, smtplib.SMTPException) as exc:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="email",
                status="error",
                payload={"error": str(exc), "subject": subject},
            )
        result = {"success": False, "error": str(exc)}
        if isinstance(exc, ValueError) or isinstance(exc, smtplib.SMTPAuthenticationError):
            result["setup_required"] = True
        return result
    except Exception as exc:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="email",
                status="error",
                payload={"error": str(exc), "subject": subject},
            )
        return {"success": False, "error": f"Failed to send email: {str(exc)}"}


def _dispatch_async(
    runner: Callable[[], Dict[str, Any]],
    description: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    result = runner()

    if result.get("success"):
        logger.info(
            "%s succeeded: %s",
            description,
            {
                key: result.get(key)
                for key in ("to", "message_id", "status", "template")
                if result.get(key) is not None
            },
        )
    else:
        logger.error("%s failed: %s", description, result)

    return result


def _record_dispatch_event(
    *,
    correlation_id: str,
    channel: str,
    status: str,
    payload: Dict[str, Any],
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        event = CommunicationEvent(
            id=str(uuid.uuid4()),
            correlation_id=correlation_id,
            channel=channel,
            event_type="dispatch",
            status=status,
            payload=payload,
        )
        record_event_sync(event)
    else:
        loop.create_task(
            comm_events.record_dispatch_async(
                correlation_id=correlation_id,
                channel=channel,
                status=status,
                payload=payload,
            )
        )



def send_email(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    html: bool = False,
) -> Dict[str, Any]:
    """Send an email or schedule it when running inside an event loop."""
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="email",
        status="scheduled",
        payload={
            "to": to,
            "subject": subject,
        },
    )

    return _dispatch_async(
        lambda: _send_email_blocking(
            to,
            subject,
            body,
            cc,
            bcc,
            html,
            correlation_id=correlation_id,
        ),
        "Email",
        {
            "to": to,
            "subject": subject,
            "correlation_id": correlation_id,
            "channel": "email",
        },
    )


def send_html_email(
    to: str,
    subject: str,
    html_body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
) -> Dict[str, Any]:
    """Send an HTML email."""
    return send_email(to, subject, html_body, cc, bcc, html=True)


def _send_bulk_email_blocking(
    recipients: str,
    subject: str,
    body: str,
    html: bool,
    *,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    recipient_list = [addr.strip() for addr in recipients.split(",") if addr.strip()]

    if not recipient_list:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="email",
                status="error",
                payload={"error": "No recipients provided", "subject": subject},
            )
        return {"success": False, "error": "No recipients provided"}

    config = get_smtp_config()
    sent_count = 0
    failed = []

    timeout = float(os.getenv("SMTP_TIMEOUT", "15"))
    with smtplib.SMTP(config["host"], config["port"], timeout=timeout) as server:
        server.starttls()
        server.login(config["user"], config["password"])

        for recipient in recipient_list:
            try:
                msg = MIMEMultipart()
                msg["From"] = config["from_email"]
                msg["To"] = recipient
                msg["Subject"] = subject
                mime_type = "html" if html else "plain"
                msg.attach(MIMEText(body, mime_type))
                try:
                    server.send_message(msg, to_addrs=[recipient])
                except TypeError:
                    server.send_message(msg)
                sent_count += 1
            except Exception as exc:
                failed.append({"email": recipient, "error": str(exc)})

    status = "sent" if not failed else "partial"
    result = {
        "success": True,
        "sent_count": sent_count,
        "total_count": len(recipient_list),
        "failed": failed,
        "message": f"Bulk email sent to {sent_count}/{len(recipient_list)} recipients",
        "correlation_id": correlation_id,
        "channel": "email",
    }

    if correlation_id:
        comm_events.record_update_sync(
            correlation_id=correlation_id,
            channel="email",
            status=status,
            payload={
                "subject": subject,
                "sent_count": sent_count,
                "total_count": len(recipient_list),
                "failed": failed,
            },
        )

    return result


def send_bulk_email(
    recipients: str,
    subject: str,
    body: str,
    html: bool = False,
) -> Dict[str, Any]:
    """Send a bulk email to comma-separated recipients."""
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="email",
        status="scheduled",
        payload={
            "recipients": recipients,
            "subject": subject,
            "html": html,
        },
    )

    return _dispatch_async(
        lambda: _send_bulk_email_blocking(
            recipients,
            subject,
            body,
            html,
            correlation_id=correlation_id,
        ),
        "Bulk email",
        {
            "recipients": recipients,
            "subject": subject,
            "correlation_id": correlation_id,
            "channel": "email",
        },
    )


def _load_email_templates() -> Dict[str, Any]:
    """Load email templates from JSON file."""
    try:
        template_path = Path(__file__).parent.parent / "data" / "templates.json"
        if not template_path.exists():
            logger.warning("Templates file not found at %s", template_path)
            return {}
        
        with open(template_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("email", {})
    except Exception as e:
        logger.error("Failed to load email templates: %s", e)
        return {}


def send_email_template(
    to: str,
    template: str,
    variables: Optional[str] = None,
) -> Dict[str, Any]:
    """Send an email using a predefined template with variable substitution."""
    templates = _load_email_templates()
    
    # Fallback to hardcoded English templates if JSON load fails or template missing
    if not templates:
        templates = {
            "order_confirmation": {
                "subject": "Order Confirmation - #{order_id}",
                "body": (
                    "Hello {name},\n\n"
                    "Your order #{order_id} has been confirmed!\n\n"
                    "We'll notify you when it ships.\n\n"
                    "Thank you for your business!"
                ),
            },
            "appointment_reminder": {
                "subject": "Appointment Reminder - {date}",
                "body": (
                    "Hello {name},\n\n"
                    "This is a reminder about your appointment:\n\n"
                    "Date: {date}\nTime: {time}\nLocation: {location}\n\n"
                    "See you soon!"
                ),
            },
            "welcome": {
                "subject": "Welcome to {company}!",
                "body": (
                    "Hello {name},\n\n"
                    "Welcome to {company}! We're excited to have you.\n\n"
                    "If you have any questions, feel free to reach out.\n\n"
                    "Best regards,\nThe Team"
                ),
            },
            "invoice": {
                "subject": "Invoice #{invoice_id} - {company}",
                "body": (
                    "Hello {name},\n\n"
                    "Please find your invoice below:\n\n"
                    "Invoice ID: {invoice_id}\nAmount: {amount}\nDue Date: {due_date}\n\n"
                    "Thank you for your business!"
                ),
            },
        }

    if template not in templates:
        return {
            "error": f"Unknown template: {template}",
            "available_templates": list(templates.keys()),
        }

    try:
        var_dict: Dict[str, Any] = json.loads(variables) if variables else {}
        tmpl = templates[template]
        subject = tmpl["subject"].format(**var_dict)
        body = tmpl["body"].format(**var_dict)
        return send_email(to=to, subject=subject, body=body)
    except KeyError as exc:
        return {"error": f"Missing required template variable: {exc}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in variables parameter"}
    except Exception as exc:
        return {"error": f"Failed to send template email: {exc}"}


async def send_email_async(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    html: bool = False,
) -> Dict[str, Any]:
    """Async wrapper for `send_email` using `asyncio.to_thread`."""
    return await asyncio.to_thread(send_email, to, subject, body, cc, bcc, html)


async def send_html_email_async(
    to: str,
    subject: str,
    html_body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_html_email`."""
    return await asyncio.to_thread(send_html_email, to, subject, html_body, cc, bcc)


async def send_bulk_email_async(
    recipients: str,
    subject: str,
    body: str,
    html: bool = False,
) -> Dict[str, Any]:
    """Async wrapper for `send_bulk_email`."""
    return await asyncio.to_thread(send_bulk_email, recipients, subject, body, html)


async def send_email_template_async(
    to: str,
    template: str,
    variables: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_email_template`."""
    return await asyncio.to_thread(send_email_template, to, template, variables)


# ══════════════════════════════════════════════════════════════════════════════
# Fire-and-Forget Handlers
# ══════════════════════════════════════════════════════════════════════════════
# These handlers return immediately and process in background.
# The agent can confirm "Email sendo enviado" without waiting 15-20s for SMTP.

async def _fire_and_forget_task(
    coro_func,
    args: tuple,
    description: str,
) -> None:
    """Execute a coroutine in background and log result."""
    try:
        result = await coro_func(*args)
        if result.get("success"):
            logger.info(
                "[Fire-and-Forget] %s succeeded: to=%s",
                description,
                result.get("to", "unknown"),
            )
        else:
            logger.error(
                "[Fire-and-Forget] %s failed: %s",
                description,
                result.get("error", "Unknown error"),
            )
    except Exception as exc:
        logger.exception("[Fire-and-Forget] %s exception: %s", description, exc)


async def send_email_fire_and_forget(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    html: bool = False,
) -> Dict[str, Any]:
    """
    Fire-and-forget email sender.
    
    Returns immediately with 'queued' status while email sends in background.
    The agent can confirm the action without waiting for SMTP.
    """
    # Create background task
    asyncio.create_task(
        _fire_and_forget_task(
            send_email_async,
            (to, subject, body, cc, bcc, html),
            f"Email to {to}",
        )
    )
    
    return {
        "success": True,
        "status": "queued",
        "message": f"Email para {to} está sendo enviado",
        "to": to,
        "subject": subject,
    }


async def send_html_email_fire_and_forget(
    to: str,
    subject: str,
    html_body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
) -> Dict[str, Any]:
    """Fire-and-forget HTML email sender."""
    asyncio.create_task(
        _fire_and_forget_task(
            send_html_email_async,
            (to, subject, html_body, cc, bcc),
            f"HTML Email to {to}",
        )
    )
    
    return {
        "success": True,
        "status": "queued",
        "message": f"Email HTML para {to} está sendo enviado",
        "to": to,
        "subject": subject,
    }


async def send_bulk_email_fire_and_forget(
    recipients: str,
    subject: str,
    body: str,
    html: bool = False,
) -> Dict[str, Any]:
    """Fire-and-forget bulk email sender."""
    recipient_count = len([r.strip() for r in recipients.split(",") if r.strip()])
    
    asyncio.create_task(
        _fire_and_forget_task(
            send_bulk_email_async,
            (recipients, subject, body, html),
            f"Bulk Email to {recipient_count} recipients",
        )
    )
    
    return {
        "success": True,
        "status": "queued",
        "message": f"Emails para {recipient_count} destinatários estão sendo enviados",
        "recipients": recipients,
        "subject": subject,
    }


async def send_email_template_fire_and_forget(
    to: str,
    template: str,
    variables: Optional[str] = None,
) -> Dict[str, Any]:
    """Fire-and-forget template email sender."""
    asyncio.create_task(
        _fire_and_forget_task(
            send_email_template_async,
            (to, template, variables),
            f"Template Email ({template}) to {to}",
        )
    )
    
    return {
        "success": True,
        "status": "queued",
        "message": f"Email de template '{template}' para {to} está sendo enviado",
        "to": to,
        "template": template,
    }


# Tool definitions for MCP
# NOTE: Using fire-and-forget handlers for instant response to the agent.
# The email is queued and sent in background while agent continues conversation.
TOOL_DEFINITIONS = [
    {
        "name": "send_email",
        "description": (
            "Send an email via SMTP with optional HTML content and CC/BCC "
            "recipients. Returns immediately while email sends in background."
        ),
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient email address (comma-separated for multiple recipients)",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line",
            },
            "body": {
                "type": "string",
                "description": "Email body content (plain text or HTML if html=True)",
            },
            "cc": {
                "type": "string",
                "description": "CC recipients (comma-separated, optional)",
            },
            "bcc": {
                "type": "string",
                "description": "BCC recipients (comma-separated, optional)",
            },
            "html": {
                "type": "boolean",
                "description": "Set to true to send HTML email (default: false)",
            },
        },
        "required": ["to", "subject", "body"],
        "handler": send_email_fire_and_forget,  # Fire-and-forget for instant response
    },
    {
        "name": "send_html_email",
        "description": "Send an HTML-formatted email with styling and formatting via SMTP. Returns immediately while email sends in background.",
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient email address (comma-separated for multiple)",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line",
            },
            "html_body": {
                "type": "string",
                "description": "HTML email body content with tags like <h1>, <p>, <b>, etc.",
            },
            "cc": {
                "type": "string",
                "description": "CC recipients (comma-separated, optional)",
            },
            "bcc": {
                "type": "string",
                "description": "BCC recipients (comma-separated, optional)",
            },
        },
        "required": ["to", "subject", "html_body"],
        "handler": send_html_email_fire_and_forget,  # Fire-and-forget for instant response
    },
    {
        "name": "send_bulk_email",
        "description": "Send the same email to multiple recipients at once (bulk/broadcast email). Returns immediately while emails send in background.",
        "parameters": {
            "recipients": {
                "type": "string",
                "description": "Comma-separated list of email addresses to send to",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line",
            },
            "body": {
                "type": "string",
                "description": "Email body content (same for all recipients)",
            },
            "html": {
                "type": "boolean",
                "description": "Set to true for HTML email (default: false)",
            },
        },
        "required": ["recipients", "subject", "body"],
        "handler": send_bulk_email_fire_and_forget,  # Fire-and-forget for instant response
    },
    {
        "name": "send_email_template",
        "description": (
            "Send an email using a predefined template (order_confirmation, "
            "appointment_reminder, welcome, or invoice) with variable substitution. "
            "Returns immediately while email sends in background."
        ),
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient email address",
            },
            "template": {
                "type": "string",
                "description": (
                    "Template name: 'order_confirmation', 'appointment_reminder', "
                    "'welcome', or 'invoice'"
                ),
            },
            "variables": {
                "type": "string",
                "description": (
                    "JSON string with template variables (e.g., '{\"name\": \"John\", "
                    "\"order_id\": \"12345\"}')"
                ),
            },
        },
        "required": ["to", "template"],
        "handler": send_email_template_fire_and_forget,  # Fire-and-forget for instant response
    },
]
