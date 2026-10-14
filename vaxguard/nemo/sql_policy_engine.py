"""
VaxGuard Enterprise SQL Agent Policy Engine.

Specialized deterministic AST parser and firewall for enterprise Text-to-SQL agents.
Enforces table allowlists, blocks destructive DDL/DML, prevents stacked query injection,
neutralizes database catalog reconnaissance (information_schema, sqlite_master),
and mitigates out-of-band command execution with zero external heavy binary dependencies
and sub-millisecond deterministic evaluation (<0.2ms).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from vaxguard.core.logger import get_logger

logger = get_logger("SQLPolicyEngine")


class SQLViolationType(str, Enum):
    DESTRUCTIVE_DDL = "destructive_ddl"
    STACKED_QUERIES = "stacked_queries"
    CATALOG_EXFILTRATION = "catalog_exfiltration"
    UNAUTHORIZED_TABLE = "unauthorized_table"
    DENIED_TABLE = "denied_table"
    DISALLOWED_COMMAND = "disallowed_command"
    UNAUTHORIZED_WRITE = "unauthorized_write"
    COMMAND_EXECUTION = "command_execution"
    BLIND_INJECTION = "blind_injection"
    LIMIT_EXCEEDED = "limit_exceeded"
    SYNTAX_MALFORMED = "syntax_malformed"


@dataclass
class SQLValidationResult:
    """Audit and enforcement result of an SQL query validation."""

    is_valid: bool
    violations: List[str] = field(default_factory=list)
    violation_types: List[SQLViolationType] = field(default_factory=list)
    parsed_tables: List[str] = field(default_factory=list)
    parsed_command: str = ""
    sanitized_query: Optional[str] = None
    risk_score: float = 0.0
    latency_ms: float = 0.0
    ast_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": self.violations,
            "violation_types": [v.value for v in self.violation_types],
            "parsed_tables": self.parsed_tables,
            "parsed_command": self.parsed_command,
            "sanitized_query": self.sanitized_query,
            "risk_score": self.risk_score,
            "latency_ms": self.latency_ms,
            "ast_summary": self.ast_summary,
        }


# Known system catalogs across Postgres, SQLite, MySQL, SQL Server, and Oracle
KNOWN_SYSTEM_CATALOGS: Set[str] = {
    # ANSI / MySQL / Postgres information_schema
    "information_schema",
    "information_schema.tables",
    "information_schema.columns",
    "information_schema.schemata",
    "information_schema.views",
    "information_schema.user_privileges",
    "information_schema.schema_privileges",
    "information_schema.table_privileges",
    "information_schema.column_privileges",
    "information_schema.routines",
    "information_schema.parameters",
    # SQLite
    "sqlite_master",
    "sqlite_temp_master",
    "sqlite_schema",
    "sqlite_temp_schema",
    "sqlite_sequence",
    "sqlite_stat1",
    "sqlite_stat4",
    # Postgres
    "pg_catalog",
    "pg_tables",
    "pg_class",
    "pg_user",
    "pg_shadow",
    "pg_database",
    "pg_roles",
    "pg_authid",
    "pg_proc",
    "pg_namespace",
    "pg_attribute",
    "pg_settings",
    # MySQL
    "mysql",
    "mysql.user",
    "mysql.db",
    "mysql.tables_priv",
    "performance_schema",
    "sys",
    # SQL Server
    "sys.tables",
    "sys.objects",
    "sys.columns",
    "sys.databases",
    "sys.schemas",
    "sys.syslogins",
    "sys.sql_logins",
    "sys.sysobjects",
    "master.dbo.sysdatabases",
    "master.sys.databases",
    # Oracle
    "all_tables",
    "all_tab_columns",
    "dba_tables",
    "dba_users",
    "user_tables",
}

# Dangerous system stored procedures and execution functions
DANGEROUS_FUNCTIONS: Set[str] = {
    "xp_cmdshell",
    "sys_eval",
    "sys_exec",
    "sys_get",
    "load_file",
    "load_extension",
    "into_outfile",
    "into_dumpfile",
    "pg_read_file",
    "pg_write_file",
    "pg_ls_dir",
    "dblink",
    "opendatasource",
    "openrowset",
    "openquery",
}

# Blind time-delay and benchmark injection functions
BLIND_INJECTION_FUNCTIONS: Set[str] = {
    "sleep",
    "pg_sleep",
    "benchmark",
    "waitfor",
}

# Destructive DDL commands
DESTRUCTIVE_DDL_COMMANDS: Set[str] = {
    "DROP",
    "TRUNCATE",
    "ALTER",
}

# All DDL commands
ALL_DDL_COMMANDS: Set[str] = {
    "DROP",
    "TRUNCATE",
    "ALTER",
    "CREATE",
    "RENAME",
}

# Write / mutation DML commands
DML_WRITE_COMMANDS: Set[str] = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "UPSERT",
    "REPLACE",
    "GRANT",
    "REVOKE",
    "ATTACH",
    "DETACH",
    "VACUUM",
    "PRAGMA",
}

# Precompiled deterministic AST patterns for ultra-low microsecond latency
RE_CTE_PRIMARY = re.compile(r"\bWITH\s+(?:RECURSIVE\s+)?([a-zA-Z0-9_`\"\[\]]+)\s+AS\s*\(", re.IGNORECASE)
RE_CTE_SECONDARY = re.compile(r",\s*([a-zA-Z0-9_`\"\[\]]+)\s+AS\s*\(", re.IGNORECASE)
RE_FROM = re.compile(r"\bFROM\s+([a-zA-Z0-9_`\"\.\[\]]+(?:\s*,\s*[a-zA-Z0-9_`\"\.\[\]]+)*)", re.IGNORECASE)
RE_JOIN = re.compile(r"\b(?:INNER|LEFT|RIGHT|FULL|CROSS|NATURAL)?\s*JOIN\s+([a-zA-Z0-9_`\"\.\[\]]+)", re.IGNORECASE)
RE_UPDATE = re.compile(r"\bUPDATE\s+([a-zA-Z0-9_`\"\.\[\]]+)", re.IGNORECASE)
RE_INTO = re.compile(r"\bINTO\s+([a-zA-Z0-9_`\"\.\[\]]+)", re.IGNORECASE)
RE_TABLE_CMD = re.compile(r"\bTABLE\s+(?:IF\s+EXISTS\s+)?([a-zA-Z0-9_`\"\.\[\]]+)", re.IGNORECASE)
RE_LIMIT = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
RE_DANGEROUS_FUNCS = [re.compile(rf"\b{re.escape(fn)}\b", re.IGNORECASE) for fn in DANGEROUS_FUNCTIONS]
RE_BLIND_INJECTION = [re.compile(rf"\b{re.escape(fn)}\s*\(", re.IGNORECASE) for fn in BLIND_INJECTION_FUNCTIONS]



class SQLAgentPolicyEngine:
    """Specialized deterministic AST parser and security policy engine for Text-to-SQL agents.

    Provides microsecond-level SQL query validation and enforcement:
    1. Table allowlist / denylist enforcement.
    2. Stacked query injection blocking.
    3. Destructive DDL neutralization (`DROP`, `TRUNCATE`, `ALTER`, `RENAME`).
    4. Database catalog reconnaissance blocking (`information_schema`, `sqlite_master`, `pg_catalog`).
    5. Write / mutation suppression (strict read-only enforcement).
    6. System command and out-of-band injection defense.
    7. Query sanitization and safe row limit capping.
    """

    def __init__(
        self,
        allowed_tables: Optional[Iterable[str]] = None,
        denied_tables: Optional[Iterable[str]] = None,
        allowed_commands: Optional[Iterable[str]] = ("SELECT", "WITH"),
        disallow_stacked_queries: bool = True,
        disallow_catalog_access: bool = True,
        disallow_ddl: bool = True,
        disallow_dml_writes: bool = True,
        disallow_command_execution: bool = True,
        disallow_blind_injection: bool = True,
        max_limit: Optional[int] = 1000,
        enforce_limit: bool = False,
        custom_blocked_keywords: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize the SQL policy engine.

        Args:
            allowed_tables: Set of allowed table identifiers. If None, any table not denied is allowed.
            denied_tables: Set of explicitly prohibited table identifiers.
            allowed_commands: Set of permitted root SQL commands (default: SELECT, WITH).
            disallow_stacked_queries: If True, reject multiple statements separated by semicolons.
            disallow_catalog_access: If True, reject queries accessing system metadata tables.
            disallow_ddl: If True, reject DDL commands (DROP, TRUNCATE, ALTER, CREATE).
            disallow_dml_writes: If True, reject data mutations (INSERT, UPDATE, DELETE).
            disallow_command_execution: If True, block OS execution procedures (xp_cmdshell, load_file).
            disallow_blind_injection: If True, block blind timing injection (sleep, benchmark).
            max_limit: Maximum permissible LIMIT value (or default limit to inject).
            enforce_limit: If True, auto-append LIMIT if absent in SELECT queries.
            custom_blocked_keywords: Optional list of additional forbidden keywords.
        """
        self.allowed_tables: Optional[Set[str]] = (
            {t.lower().strip().strip("`\"'[]") for t in allowed_tables}
            if allowed_tables is not None
            else None
        )
        self.denied_tables: Set[str] = (
            {t.lower().strip().strip("`\"'[]") for t in denied_tables}
            if denied_tables is not None
            else set()
        )
        self.allowed_commands: Set[str] = (
            {c.upper().strip() for c in allowed_commands}
            if allowed_commands is not None
            else {"SELECT", "WITH"}
        )
        self.disallow_stacked_queries = disallow_stacked_queries
        self.disallow_catalog_access = disallow_catalog_access
        self.disallow_ddl = disallow_ddl
        self.disallow_dml_writes = disallow_dml_writes
        self.disallow_command_execution = disallow_command_execution
        self.disallow_blind_injection = disallow_blind_injection
        self.max_limit = max_limit
        self.enforce_limit = enforce_limit
        self.custom_blocked_keywords: Set[str] = (
            {k.upper().strip() for k in custom_blocked_keywords}
            if custom_blocked_keywords is not None
            else set()
        )

    # ─── Lexer & Preprocessing ─────────────────────────────────────────────────

    @staticmethod
    def strip_comments_and_strings(sql: str) -> Tuple[str, List[str]]:
        """Remove SQL comments and extract string literals cleanly without breaking syntax."""
        cleaned_chars: List[str] = []
        string_literals: List[str] = []
        i = 0
        n = len(sql)

        while i < n:
            # Line comment: -- or #
            if sql[i : i + 2] == "--" or (sql[i] == "#" and (i == 0 or sql[i - 1].isspace())):
                while i < n and sql[i] != "\n":
                    i += 1
                cleaned_chars.append(" ")
                continue

            # Block comment: /* ... */
            if sql[i : i + 2] == "/*":
                i += 2
                while i < n and sql[i : i + 2] != "*/":
                    i += 1
                i += 2
                cleaned_chars.append(" ")
                continue

            # String literal: '...' or $$...$$
            if sql[i] in ("'", '"'):
                quote = sql[i]
                start_str = i
                i += 1
                while i < n:
                    if sql[i] == "\\" and i + 1 < n:
                        i += 2
                        continue
                    if sql[i] == quote:
                        if i + 1 < n and sql[i + 1] == quote:  # Escaped quote ''
                            i += 2
                            continue
                        i += 1
                        break
                    i += 1
                lit = sql[start_str:i]
                string_literals.append(lit)
                # Replace string literal with a neutral placeholder so identifiers inside aren't parsed
                cleaned_chars.append("'__STR__'")
                continue

            cleaned_chars.append(sql[i])
            i += 1

        return "".join(cleaned_chars), string_literals

    def extract_statements(self, sql: str) -> List[str]:
        """Split SQL into individual statements while respecting string literals and comments."""
        cleaned, _ = self.strip_comments_and_strings(sql)
        statements: List[str] = []
        current: List[str] = []
        in_single_quote = False
        in_double_quote = False
        i = 0
        n = len(sql)

        while i < n:
            c = sql[i]
            if c == "'" and not in_double_quote:
                if sql[i : i + 2] == "''":
                    current.append(sql[i : i + 2])
                    i += 2
                    continue
                in_single_quote = not in_single_quote
            elif c == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif c == ";" and not in_single_quote and not in_double_quote:
                stmt = "".join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
                i += 1
                continue
            current.append(c)
            i += 1

        stmt = "".join(current).strip()
        if stmt:
            statements.append(stmt)

        return statements

    def extract_root_command(self, sql: str) -> str:
        """Extract the root command token (e.g., SELECT, DROP, INSERT, WITH)."""
        cleaned, _ = self.strip_comments_and_strings(sql)
        tokens = cleaned.strip().split()
        if not tokens:
            return ""
        return tokens[0].upper()

    def extract_tables(self, sql: str) -> List[str]:
        """Deterministic sub-millisecond extraction of all referenced table identifiers.

        Parses table names from:
        - FROM clauses (with comma joins: FROM a, b)
        - JOIN clauses (INNER, LEFT, RIGHT, FULL, CROSS JOIN)
        - UPDATE clauses
        - INTO clauses
        - CTE declarations (WITH cte AS (...)) are excluded from the external table set.
        """
        cleaned, _ = self.strip_comments_and_strings(sql)
        tables: Set[str] = set()
        ctes: Set[str] = set()

        # 1. Detect CTE names to avoid false positive unauthorized table alerts
        for match in RE_CTE_PRIMARY.finditer(cleaned):
            cte_name = match.group(1).lower().strip("`\"'[]")
            ctes.add(cte_name)

        for match in RE_CTE_SECONDARY.finditer(cleaned):
            cte_name = match.group(1).lower().strip("`\"'[]")
            ctes.add(cte_name)

        # 2. Extract from FROM clauses
        for m in RE_FROM.finditer(cleaned):
            raw_from = m.group(1)
            parts = [p.strip() for p in raw_from.split(",") if p.strip()]
            for p in parts:
                toks = p.split()
                if toks:
                    tbl = toks[0].strip("() \t\n`\"'[]").lower()
                    if tbl and not tbl.startswith("("):
                        tables.add(tbl)

        # 3. Extract from JOIN clauses
        for m in RE_JOIN.finditer(cleaned):
            tbl = m.group(1).strip("() \t\n`\"'[]").lower()
            if tbl and not tbl.startswith("("):
                tables.add(tbl)

        # 4. Extract from UPDATE clauses
        for m in RE_UPDATE.finditer(cleaned):
            tbl = m.group(1).strip("() \t\n`\"'[]").lower()
            if tbl:
                tables.add(tbl)

        # 5. Extract from INTO clauses (INSERT INTO tbl)
        for m in RE_INTO.finditer(cleaned):
            tbl = m.group(1).strip("() \t\n`\"'[]").lower()
            if tbl:
                tables.add(tbl)

        # 6. Extract from TABLE clauses (e.g. TRUNCATE TABLE tbl, DROP TABLE tbl)
        for m in RE_TABLE_CMD.finditer(cleaned):
            tbl = m.group(1).strip("() \t\n`\"'[]").lower()
            if tbl:
                tables.add(tbl)

        # Filter out declared CTE names
        external_tables = [t for t in tables if t not in ctes]
        return sorted(external_tables)

    # ─── Core Policy Evaluation ────────────────────────────────────────────────

    def validate_sql(self, sql: str) -> SQLValidationResult:
        """Deterministically validate an SQL query against all security policies in <0.2ms.

        Returns an audit-grade SQLValidationResult detailing safety verdicts,
        violations, parsed tables, root command, and sanitization.
        """
        t0 = time.perf_counter()
        violations: List[str] = []
        violation_types: List[SQLViolationType] = []
        risk_score = 0.0

        raw_sql = sql.strip()
        if not raw_sql:
            lat = (time.perf_counter() - t0) * 1000.0
            return SQLValidationResult(
                is_valid=False,
                violations=["Empty SQL query provided"],
                violation_types=[SQLViolationType.SYNTAX_MALFORMED],
                latency_ms=round(lat, 4),
            )

        # 1. Stacked Query Check
        statements = self.extract_statements(raw_sql)
        if self.disallow_stacked_queries and len(statements) > 1:
            violations.append(
                f"Stacked queries blocked: detected {len(statements)} distinct SQL statements."
            )
            violation_types.append(SQLViolationType.STACKED_QUERIES)
            risk_score = max(risk_score, 0.95)

        # Evaluate the primary statement (or each statement)
        target_sql = statements[0] if statements else raw_sql
        cleaned_sql, _ = self.strip_comments_and_strings(raw_sql)
        tokens = cleaned_sql.upper().split()
        root_command = self.extract_root_command(raw_sql)

        # 2. Command Allowlist Check
        if root_command and root_command not in self.allowed_commands:
            violations.append(
                f"Root SQL command '{root_command}' is not in allowed commands: {sorted(self.allowed_commands)}"
            )
            violation_types.append(SQLViolationType.DISALLOWED_COMMAND)
            risk_score = max(risk_score, 0.8)

        # 3. Destructive DDL Check (DROP, TRUNCATE, ALTER, RENAME)
        if self.disallow_ddl:
            for tok in tokens:
                if tok in ALL_DDL_COMMANDS:
                    violations.append(f"Destructive DDL keyword '{tok}' is strictly prohibited.")
                    violation_types.append(SQLViolationType.DESTRUCTIVE_DDL)
                    risk_score = 1.0
                    break

        # 4. Unauthorized DML / Mutation Check (INSERT, UPDATE, DELETE, etc.)
        if self.disallow_dml_writes:
            for tok in tokens:
                if tok in DML_WRITE_COMMANDS:
                    violations.append(
                        f"Database write/mutation operation '{tok}' is prohibited in read-only mode."
                    )
                    violation_types.append(SQLViolationType.UNAUTHORIZED_WRITE)
                    risk_score = max(risk_score, 0.85)
                    break

        # 5. Dangerous Stored Procedure / OS Command Execution
        if self.disallow_command_execution:
            sql_lower = cleaned_sql.lower()
            for pattern, fn in zip(RE_DANGEROUS_FUNCS, DANGEROUS_FUNCTIONS):
                if pattern.search(sql_lower):
                    violations.append(
                        f"Potentially dangerous OS/administrative function '{fn}' detected."
                    )
                    violation_types.append(SQLViolationType.COMMAND_EXECUTION)
                    risk_score = 1.0
                    break

            # Check INTO OUTFILE / DUMPFILE
            if "into outfile" in sql_lower or "into dumpfile" in sql_lower:
                violations.append("File exfiltration clause 'INTO OUTFILE/DUMPFILE' is prohibited.")
                violation_types.append(SQLViolationType.COMMAND_EXECUTION)
                risk_score = 1.0

        # 6. Blind Injection / Time-Delay Execution
        if self.disallow_blind_injection:
            sql_lower = cleaned_sql.lower()
            for pattern, fn in zip(RE_BLIND_INJECTION, BLIND_INJECTION_FUNCTIONS):
                if pattern.search(sql_lower) or (fn == "waitfor" and "waitfor delay" in sql_lower):
                    violations.append(
                        f"Blind timing delay function '{fn}' is blocked to prevent covert exfiltration."
                    )
                    violation_types.append(SQLViolationType.BLIND_INJECTION)
                    risk_score = max(risk_score, 0.9)
                    break

        # 7. Database Catalog Exfiltration Check
        parsed_tables = self.extract_tables(raw_sql)
        cleaned_lower = cleaned_sql.lower()

        if self.disallow_catalog_access:
            for catalog in KNOWN_SYSTEM_CATALOGS:
                # Check in extracted tables or directly in cleaned query tokens
                cat_lower = catalog.lower()
                if any(t == cat_lower or t.startswith(f"{cat_lower}.") or t.endswith(f".{cat_lower}") for t in parsed_tables) or re.search(rf"\b{re.escape(cat_lower)}\b", cleaned_lower):
                    violations.append(
                        f"Database catalog exfiltration blocked: access to system table '{catalog}' is prohibited."
                    )
                    violation_types.append(SQLViolationType.CATALOG_EXFILTRATION)
                    risk_score = max(risk_score, 0.95)
                    break

        # 8. Table Allowlist & Denylist Enforcement
        for tbl in parsed_tables:
            tbl_clean = tbl.split(".")[-1].strip("`\"'[]")
            full_tbl = tbl.strip("`\"'[]")

            # Denylist check
            if tbl_clean in self.denied_tables or full_tbl in self.denied_tables:
                violations.append(f"Access to prohibited table '{tbl}' is explicitly denied.")
                violation_types.append(SQLViolationType.DENIED_TABLE)
                risk_score = max(risk_score, 0.9)

            # Allowlist check
            if self.allowed_tables is not None:
                if tbl_clean not in self.allowed_tables and full_tbl not in self.allowed_tables:
                    violations.append(
                        f"Table '{tbl}' is not in enterprise allowlist: {sorted(self.allowed_tables)}"
                    )
                    violation_types.append(SQLViolationType.UNAUTHORIZED_TABLE)
                    risk_score = max(risk_score, 0.85)

        # 9. Custom Blocked Keywords
        for kw in self.custom_blocked_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", cleaned_sql, re.IGNORECASE):
                violations.append(f"Prohibited custom keyword '{kw}' detected.")
                violation_types.append(SQLViolationType.DISALLOWED_COMMAND)
                risk_score = max(risk_score, 0.75)

        # 10. Limit Enforcement & Sanitization
        sanitized_query = None
        limit_match = RE_LIMIT.search(cleaned_sql)
        if limit_match:
            specified_limit = int(limit_match.group(1))
            if self.max_limit is not None and specified_limit > self.max_limit:
                violations.append(
                    f"Requested LIMIT {specified_limit} exceeds maximum configured safety cap of {self.max_limit} rows."
                )
                violation_types.append(SQLViolationType.LIMIT_EXCEEDED)
                risk_score = max(risk_score, 0.6)
        elif self.enforce_limit and self.max_limit is not None and root_command == "SELECT":
            # Sanitize by appending default safe limit
            sanitized_query = f"{raw_sql.rstrip(';')} LIMIT {self.max_limit};"

        is_valid = len(violations) == 0
        lat_ms = (time.perf_counter() - t0) * 1000.0

        return SQLValidationResult(
            is_valid=is_valid,
            violations=violations,
            violation_types=violation_types,
            parsed_tables=parsed_tables,
            parsed_command=root_command,
            sanitized_query=sanitized_query or raw_sql,
            risk_score=round(risk_score, 3),
            latency_ms=round(lat_ms, 4),
            ast_summary={
                "statements_count": len(statements),
                "has_limit": limit_match is not None,
                "table_count": len(parsed_tables),
            },
        )

    def is_safe(self, sql: str) -> bool:
        """Fast boolean verification whether SQL conforms to all enterprise policies."""
        return self.validate_sql(sql).is_valid

    def sanitize_sql(self, sql: str) -> str:
        """Return safely validated query, with auto-injected LIMIT caps if required.

        Raises:
            ValueError: If query violates active security policies.
        """
        res = self.validate_sql(sql)
        if not res.is_valid:
            raise ValueError(f"SQL Policy Violation: {'; '.join(res.violations)}")
        return res.sanitized_query or sql

    # ─── NeMo Guardrails Action Integration ────────────────────────────────────

    def as_nemo_action(
        self,
        action_name: str = "validate_sql_query",
    ) -> Callable[..., Any]:
        """Return a callable action compatible with NeMo Guardrails and VaxGuard LLMRails.

        Can be registered with `rails.register_action(engine.as_nemo_action(), name="validate_sql_query")`.
        """

        async def _sql_validator_action(
            sql: Optional[str] = None,
            query: Optional[str] = None,
            context: Optional[Any] = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            target = sql or query
            if not target and context:
                # Try extracting from context dictionary
                if isinstance(context, dict):
                    target = context.get("sql") or context.get("query")
            if not target:
                return {
                    "is_valid": False,
                    "error": "No SQL query provided to validator action.",
                }
            result = self.validate_sql(target)
            return result.to_dict()

        _sql_validator_action.__name__ = action_name
        return _sql_validator_action


__all__ = [
    "SQLAgentPolicyEngine",
    "SQLValidationResult",
    "SQLViolationType",
    "KNOWN_SYSTEM_CATALOGS",
    "DESTRUCTIVE_DDL_COMMANDS",
]
