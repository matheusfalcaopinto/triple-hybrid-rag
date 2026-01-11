import logging
from typing import Optional

from supabase import Client, create_client

from voice_agent.config import SETTINGS

logger = logging.getLogger(__name__)

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Get or initialize the Supabase client.
    """
    global _supabase_client

    if _supabase_client is None:
        url = SETTINGS.supabase_url
        key = SETTINGS.supabase_service_role_key

        if not url or not key:
            logger.warning("Supabase URL or Key not set. DB operations will fail.")
            # We return a client anyway, but it might fail on requests if url is empty.
            # Ideally we should raise error, but for now let's allow app to start.
        
        try:
            _supabase_client = create_client(url, key)
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    return _supabase_client
