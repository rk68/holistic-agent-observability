from __future__ import annotations

import ast
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Final

from langchain_core.tools import tool

from .state import get_current_state
from .visibility import (
    ArtefactInstance,
    DataArtefactKind,
    DataSensitivity,
    get_current_tracker,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DATA_ROOT: Final[Path] = _PROJECT_ROOT / "data"
_BANKING_DATA_PATH: Final[Path] = _DATA_ROOT / "banking" / "banking_state.json"
_BANKING_DB_PATH: Final[Path] = _DATA_ROOT / "banking" / "banking.db"
_ALLOWED_AST_NODES: Final[tuple[type[ast.AST], ...]] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.FloorDiv,
    ast.Constant,
    ast.Call,
    ast.Name,
)
_SAFE_FUNCTIONS: Final[dict[str, Callable[..., float | int]]] = {
    "abs": abs,
    "round": round,
}


def _map_sensitivity_level(value: DataSensitivity | None) -> str:
    if value is DataSensitivity.HIGH:
        return "HIGHLY_SENSITIVE"
    if value is DataSensitivity.MEDIUM:
        return "SENSITIVE"
    if value is DataSensitivity.LOW:
        return "INTERNAL"
    return "PUBLIC"


def _record_data_artefact(
    *,
    instance: ArtefactInstance,
    source: str,
    payload: dict[str, Any],
    fields_sensitivity: dict[str, str] | None = None,
) -> None:
    state = get_current_state()
    if state is None:
        return

    data_context = state.setdefault("data_context", {})
    level = _map_sensitivity_level(instance.sensitivity)
    artefact: dict[str, Any] = {
        "id": instance.id,
        "source": source,
        "payload": payload,
        "sensitivity": level,
        "fields_sensitivity": dict(fields_sensitivity or {}),
    }
    data_context[instance.id] = artefact


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported operation: {ast.dump(node, maxlen=40)}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
                raise ValueError("Only abs() and round() are permitted in calculations")
    compiled = compile(tree, filename="<calculator>", mode="eval")
    return float(eval(compiled, {"__builtins__": {}}, _SAFE_FUNCTIONS))


@tool("math.calculator", return_direct=False)
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Supports +, -, *, /, %, //, and ** operators along with abs() and round().
    """

    solution = _safe_eval(expression)
    result = str(solution)

    tracker = get_current_tracker()
    if tracker is not None:
        instance = tracker.register_tool_result(
            tool_name="math.calculator",
            kind=DataArtefactKind.TOOL_OUTPUT,
            sensitivity=DataSensitivity.LOW,
            tags=["math"],
        )

        _record_data_artefact(
            instance=instance,
            source="tool:math.calculator",
            payload={"expression": expression, "result": result},
            fields_sensitivity={"expression": "INTERNAL", "result": "INTERNAL"},
        )

    return result


@tool("project.read_file", return_direct=False)
def read_project_file(relative_path: str) -> str:
    """Read a UTF-8 encoded file relative to the project root.

    The path must stay within the repository.
    """

    path = (_PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise FileNotFoundError("File is outside the project root") from exc

    if not path.is_file():
        raise FileNotFoundError("File is outside the project or does not exist")

    contents = path.read_text(encoding="utf-8")

    tracker = get_current_tracker()
    if tracker is not None:
        instance = tracker.register_tool_result(
            tool_name="project.read_file",
            kind=DataArtefactKind.FILE_CONTENTS,
            sensitivity=DataSensitivity.MEDIUM,
            tags=["file"],
        )

        _record_data_artefact(
            instance=instance,
            source="tool:project.read_file",
            payload={"path": str(path), "preview": contents[:2000]},
            fields_sensitivity={"path": "INTERNAL", "preview": "SENSITIVE"},
        )

    return contents


def _parse_iso8601(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


@lru_cache(maxsize=1)
def _load_banking_state() -> dict[str, Any]:
    if not _BANKING_DATA_PATH.exists():
        raise FileNotFoundError(
            "Banking data file not found. Ensure agent/data/banking_state.json exists."
        )
    return json.loads(_BANKING_DATA_PATH.read_text(encoding="utf-8"))


def _resolve_account(identifier: str) -> dict[str, Any]:
    state = _load_banking_state()
    needle = identifier.strip().lower()
    for account in state.get("accounts", []):
        if account["account_id"].lower() == needle:
            return account
        if account.get("display_name", "").lower() == needle:
            return account
        if account.get("type", "").lower() == needle:
            return account
    raise ValueError(f"Unknown account: {identifier}")


@tool("banking.get_account_balance", return_direct=False)
def get_account_balance(account_identifier: str) -> str:
    """Return the current balance for a customer account.

    Provide either the account_id (e.g. "acct-chk-001") or a human label like
    "Everyday Checking"/"checking".
    """

    account = _resolve_account(account_identifier)
    currency = account.get("currency", "USD")
    if account.get("type") == "credit":
        remaining = account.get("remaining_balance", 0.0)
        limit_amount = account.get("credit_limit")
        utilization = None
        if limit_amount:
            utilization = (remaining / limit_amount) * 100
        parts = [
            f"Credit balance for {account['display_name']}: {remaining:,.2f} {currency}",
        ]
        if limit_amount is not None:
            parts.append(f"Credit limit: {limit_amount:,.2f} {currency}")
        if utilization is not None:
            parts.append(f"Utilization: {utilization:.1f}%")
        result = "\n".join(parts)
    else:
        balance = account.get("balance", 0.0)
        result = f"Available balance for {account['display_name']}: {balance:,.2f} {currency}"

    tracker = get_current_tracker()
    if tracker is not None:
        instance = tracker.register_tool_result(
            tool_name="banking.get_account_balance",
            kind=DataArtefactKind.TOOL_OUTPUT,
            sensitivity=DataSensitivity.HIGH,
            tags=["banking", "balance"],
        )

        _record_data_artefact(
            instance=instance,
            source="tool:banking.get_account_balance",
            payload={
                "account_identifier": account_identifier,
                "preview": result[:2000],
            },
            fields_sensitivity={
                "account_identifier": "HIGHLY_SENSITIVE",
                "preview": "HIGHLY_SENSITIVE",
            },
        )

    return result


@tool("banking.get_recent_transactions", return_direct=False)
def get_recent_transactions(
    account_identifier: str,
    *,
    lookback_days: int = 7,
    minimum_amount: float | None = None,
) -> str:
    """Summarize recent transactions for an account.

    Filters to the past `lookback_days` and optionally highlights items whose absolute
    value meets or exceeds `minimum_amount`.
    """

    account = _resolve_account(account_identifier)
    account_id = account["account_id"]
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=max(lookback_days, 1))

    state = _load_banking_state()
    transactions = [
        txn
        for txn in state.get("transactions", [])
        if txn.get("account_id") == account_id
        and _parse_iso8601(txn["timestamp"]) >= cutoff
    ]

    if minimum_amount is not None:
        highlight = [
            txn
            for txn in transactions
            if abs(float(txn.get("amount", 0.0))) >= minimum_amount
        ]
    else:
        highlight = transactions

    lines = [
        f"Recent transactions for {account['display_name']} (last {lookback_days} days):",
    ]
    if not transactions:
        lines.append("• No transactions found in the requested window.")
        result = "\n".join(lines)
    else:
        for txn in transactions:
            timestamp = _parse_iso8601(txn["timestamp"]).strftime("%Y-%m-%d %H:%M")
            amount = float(txn.get("amount", 0.0))
            tag = " 🔍" if txn in highlight and minimum_amount is not None else ""
            lines.append(
                f"• {timestamp}: {amount:,.2f} ({txn.get('category', 'uncategorized')}) - {txn.get('description', '')}{tag}"
            )

        if minimum_amount is not None and highlight:
            lines.append(
                f"Highlighted items exceed |amount| ≥ {minimum_amount:,.2f}."
            )
        result = "\n".join(lines)

    tracker = get_current_tracker()
    if tracker is not None:
        instance = tracker.register_tool_result(
            tool_name="banking.get_recent_transactions",
            kind=DataArtefactKind.TOOL_OUTPUT,
            sensitivity=DataSensitivity.HIGH,
            tags=["banking", "transactions"],
        )

        _record_data_artefact(
            instance=instance,
            source="tool:banking.get_recent_transactions",
            payload={
                "account_identifier": account_identifier,
                "lookback_days": lookback_days,
                "minimum_amount": minimum_amount,
                "preview": result[:2000],
            },
            fields_sensitivity={
                "account_identifier": "HIGHLY_SENSITIVE",
                "preview": "HIGHLY_SENSITIVE",
            },
        )

    return result


@tool("banking.recommend_products", return_direct=False)
def recommend_products(
    *,
    goal: str | None = None,
    segment: str | None = None,
    top_k: int = 3,
) -> str:
    """Recommend banking products aligned to the customer's goals or segments.

    Provide a goal keyword (e.g. "travel", "savings") or a segment such as "credit".
    """

    state = _load_banking_state()
    desired = {value.strip().lower() for value in [goal, segment] if value}

    products = state.get("products", [])
    if desired:
        products = [
            product
            for product in products
            if desired.intersection({tag.lower() for tag in product.get("ideal_for", [])})
        ]

    if not products:
        result = "No matching products found for the requested focus."
        tracker = get_current_tracker()
        if tracker is not None:
            instance = tracker.register_tool_result(
                tool_name="banking.recommend_products",
                kind=DataArtefactKind.TOOL_OUTPUT,
                sensitivity=DataSensitivity.MEDIUM,
                tags=["banking", "products"],
            )
            _record_data_artefact(
                instance=instance,
                source="tool:banking.recommend_products",
                payload={
                    "goal": goal,
                    "segment": segment,
                    "preview": result,
                },
                fields_sensitivity={
                    "goal": "INTERNAL",
                    "segment": "INTERNAL",
                    "preview": "INTERNAL",
                },
            )
        return result

    products.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    selection = products[: max(1, top_k)]

    lines = ["Recommended products:"]
    for product in selection:
        summary_parts = [product.get("name", "Unnamed product")]
        if product.get("type"):
            summary_parts.append(f"type: {product['type']}")
        if product.get("score") is not None:
            summary_parts.append(f"fit score: {product['score']:.2f}")
        if product.get("rewards_summary"):
            summary_parts.append(product["rewards_summary"])
        elif product.get("apy_percent") is not None:
            summary_parts.append(f"APY {product['apy_percent']:.2f}%")
        lines.append("• " + " — ".join(summary_parts))

    result = "\n".join(lines)

    tracker = get_current_tracker()
    if tracker is not None:
        instance = tracker.register_tool_result(
            tool_name="banking.recommend_products",
            kind=DataArtefactKind.TOOL_OUTPUT,
            sensitivity=DataSensitivity.MEDIUM,
            tags=["banking", "products"],
        )

        _record_data_artefact(
            instance=instance,
            source="tool:banking.recommend_products",
            payload={
                "goal": goal,
                "segment": segment,
                "preview": result[:2000],
            },
            fields_sensitivity={
                "goal": "INTERNAL",
                "segment": "INTERNAL",
                "preview": "INTERNAL",
            },
        )

    return result


@tool("banking.sql_query", return_direct=False)
def run_banking_sql(query: str) -> str:
    """Execute a read-only SQL query against the sample banking warehouse.

    The query must start with SELECT and is automatically limited to 100 rows when
    no explicit LIMIT clause is provided.
    """

    if not _BANKING_DB_PATH.exists():
        raise FileNotFoundError(
            "Banking SQLite database missing. Expected at data/banking.db"
        )

    stripped = query.strip()
    if not stripped:
        raise ValueError("Provide a SELECT statement to execute.")
    if not stripped.lower().startswith("select"):
        raise ValueError("Only read-only SELECT statements are permitted.")

    normalized = stripped.rstrip(";\n ")
    if "limit" not in normalized.lower():
        normalized = f"{normalized} LIMIT 100"

    try:
        with sqlite3.connect(_BANKING_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(normalized)
            rows = cursor.fetchmany(100)
    except sqlite3.OperationalError as exc:
        return f"SQL execution error: {exc}"

    if not rows:
        return "Query returned no rows."

    headers = rows[0].keys()
    header_line = " | ".join(headers)
    separator = " | ".join(["-" * len(col) for col in headers])
    body_lines = [" | ".join(str(row[col]) for col in headers) for row in rows]

    result = "\n".join([header_line, separator, *body_lines])

    tracker = get_current_tracker()
    if tracker is not None:
        instance = tracker.register_tool_result(
            tool_name="banking.sql_query",
            kind=DataArtefactKind.SQL_ROWSET,
            sensitivity=DataSensitivity.HIGH,
            tags=["banking", "sql"],
        )

        _record_data_artefact(
            instance=instance,
            source="tool:banking.sql_query",
            payload={
                "query": normalized,
                "preview": result[:2000],
            },
            fields_sensitivity={
                "query": "INTERNAL",
                "preview": "HIGHLY_SENSITIVE",
            },
        )

    return result


DEFAULT_TOOLS: Final[dict[str, tool]] = {
    "math": calculator,
    "read_file": read_project_file,
    "banking_balance": get_account_balance,
    "banking_transactions": get_recent_transactions,
    "banking_products": recommend_products,
    "banking_sql": run_banking_sql,
}
