"""
Thin data-access layer around the Supabase `files` table.
Uses the service_role key (server-side only — never expose this to the frontend).
"""
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE = "files"


def insert_file(record: dict) -> dict:
    result = supabase.table(TABLE).insert(record).execute()
    return result.data[0] if result.data else {}


def file_exists(file_unique_id: str) -> bool:
    if not file_unique_id:
        return False
    result = (
        supabase.table(TABLE)
        .select("id")
        .eq("file_unique_id", file_unique_id)
        .limit(1)
        .execute()
    )
    return bool(result.data)


def get_all_files() -> list[dict]:
    result = (
        supabase.table(TABLE)
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def get_files_by_ids(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    result = supabase.table(TABLE).select("*").in_("id", ids).execute()
    return result.data or []
