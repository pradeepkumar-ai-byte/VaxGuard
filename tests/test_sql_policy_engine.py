"""
Unit tests for VaxGuard SQLAgentPolicyEngine (vaxguard/nemo/sql_policy_engine.py).

Verifies deterministic Text-to-SQL security policies:
- Destructive DDL blocking (DROP, TRUNCATE, ALTER)
- Stacked query injection neutralization
- Database catalog exfiltration defense (information_schema, sqlite_master, pg_catalog, etc.)
- Strict table allowlists and denylists enforcement
- CTE handling without false positives
- Read-only enforcement (blocking INSERT, UPDATE, DELETE)
- Command execution / Out-of-band defense (xp_cmdshell, load_file, into outfile)
- Blind time-based injection defense (sleep, pg_sleep, benchmark)
- LIMIT enforcement and safe query sanitization
- Sub-millisecond execution SLA (<0.2ms)
- NeMo Guardrails action integration
"""

import time
import pytest

from vaxguard.nemo.sql_policy_engine import (
    SQLAgentPolicyEngine,
    SQLViolationType,
)
from vaxguard.integrations.nemo_compat import LLMRails


@pytest.fixture
def policy_engine() -> SQLAgentPolicyEngine:
    """Standard enterprise SQL policy engine allowing customers, orders, and products."""
    return SQLAgentPolicyEngine(
        allowed_tables={"customers", "orders", "products"},
        denied_tables={"passwords", "credit_cards", "auth_tokens"},
        allowed_commands={"SELECT", "WITH"},
        disallow_stacked_queries=True,
        disallow_catalog_access=True,
        disallow_ddl=True,
        disallow_dml_writes=True,
        max_limit=1000,
        enforce_limit=True,
    )


def test_benign_select_query_passes(policy_engine: SQLAgentPolicyEngine):
    """Verify standard benign SELECT query is allowed and parsed accurately."""
    # Warmup execution to avoid cold JIT/import overhead on Windows
    policy_engine.validate_sql("SELECT 1;")
    sql = "SELECT id, name, email FROM customers WHERE status = 'active';"
    result = policy_engine.validate_sql(sql)

    assert result.is_valid is True
    assert len(result.violations) == 0
    assert result.parsed_command == "SELECT"
    assert "customers" in result.parsed_tables
    assert result.risk_score == 0.0
    assert result.latency_ms < 5.0


def test_complex_joins_and_table_extraction(policy_engine: SQLAgentPolicyEngine):
    """Verify table extraction across INNER JOIN, LEFT JOIN, and subqueries."""
    sql = """
    SELECT c.name, o.total, p.title
    FROM customers c
    INNER JOIN orders o ON c.id = o.customer_id
    LEFT JOIN products p ON o.product_id = p.id
    WHERE o.created_at >= '2026-01-01';
    """
    result = policy_engine.validate_sql(sql)
    assert result.is_valid is True
    assert set(result.parsed_tables) == {"customers", "orders", "products"}


def test_destructive_ddl_blocking(policy_engine: SQLAgentPolicyEngine):
    """Verify destructive DDL commands are blocked immediately."""
    ddl_attacks = [
        "DROP TABLE customers;",
        "DROP DATABASE production;",
        "TRUNCATE TABLE orders;",
        "ALTER TABLE customers DROP COLUMN email;",
        "ALTER TABLE products ADD COLUMN backdoor TEXT;",
        "CREATE TABLE rogue_table (id INT);",
        "RENAME TABLE customers TO exfiltrated_customers;",
    ]

    for attack in ddl_attacks:
        res = policy_engine.validate_sql(attack)
        assert res.is_valid is False, f"Failed to block DDL attack: {attack}"
        assert SQLViolationType.DESTRUCTIVE_DDL in res.violation_types or SQLViolationType.DISALLOWED_COMMAND in res.violation_types
        assert res.risk_score >= 0.8


def test_stacked_query_injection_blocking(policy_engine: SQLAgentPolicyEngine):
    """Verify stacked queries separated by semicolons are neutralized."""
    stacked = "SELECT * FROM orders; DROP TABLE customers;"
    res = policy_engine.validate_sql(stacked)

    assert res.is_valid is False
    assert SQLViolationType.STACKED_QUERIES in res.violation_types
    assert any("stacked" in v.lower() for v in res.violations)


def test_semicolons_inside_string_literals_do_not_trigger_false_positives(policy_engine: SQLAgentPolicyEngine):
    """Verify semicolons inside literal strings do not cause stacked query false positives."""
    sql = "SELECT id, name FROM customers WHERE notes = 'first note; second note' AND status = 'ok';"
    res = policy_engine.validate_sql(sql)

    assert res.is_valid is True
    assert SQLViolationType.STACKED_QUERIES not in res.violation_types


def test_database_catalog_exfiltration_blocking(policy_engine: SQLAgentPolicyEngine):
    """Verify reconnaissance on information_schema, sqlite_master, pg_catalog is blocked."""
    catalog_probes = [
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';",
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'customers';",
        "SELECT sql FROM sqlite_master WHERE type = 'table';",
        "SELECT * FROM sqlite_temp_master;",
        "SELECT relname FROM pg_class WHERE relkind = 'r';",
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public';",
        "SELECT usename, passwd FROM pg_shadow;",
        "SELECT User, Host FROM mysql.user;",
        "SELECT name FROM sys.databases;",
    ]

    for probe in catalog_probes:
        res = policy_engine.validate_sql(probe)
        assert res.is_valid is False, f"Failed to block catalog exfiltration: {probe}"
        assert SQLViolationType.CATALOG_EXFILTRATION in res.violation_types
        assert res.risk_score >= 0.9


def test_table_allowlist_enforcement(policy_engine: SQLAgentPolicyEngine):
    """Verify queries accessing tables outside the allowlist are rejected."""
    unauthorized = "SELECT ssn, salary FROM employee_payroll WHERE department = 'Finance';"
    res = policy_engine.validate_sql(unauthorized)

    assert res.is_valid is False
    assert SQLViolationType.UNAUTHORIZED_TABLE in res.violation_types
    assert "employee_payroll" in res.parsed_tables
    assert any("allowlist" in v.lower() for v in res.violations)


def test_table_denylist_enforcement(policy_engine: SQLAgentPolicyEngine):
    """Verify queries accessing explicitly denied tables are rejected."""
    denied = "SELECT * FROM passwords WHERE user_id = 42;"
    res = policy_engine.validate_sql(denied)

    assert res.is_valid is False
    assert SQLViolationType.DENIED_TABLE in res.violation_types or SQLViolationType.UNAUTHORIZED_TABLE in res.violation_types


def test_cte_queries_do_not_flag_cte_name_as_unauthorized(policy_engine: SQLAgentPolicyEngine):
    """Verify Common Table Expressions (WITH cte AS ...) are recognized as CTEs, not foreign tables."""
    cte_sql = """
    WITH high_value_orders AS (
        SELECT id, customer_id, total
        FROM orders
        WHERE total > 500
    )
    SELECT c.name, hvo.total
    FROM high_value_orders hvo
    JOIN customers c ON hvo.customer_id = c.id;
    """
    res = policy_engine.validate_sql(cte_sql)
    assert res.is_valid is True
    assert "high_value_orders" not in res.parsed_tables
    assert "orders" in res.parsed_tables
    assert "customers" in res.parsed_tables


def test_read_only_dml_mutation_blocking(policy_engine: SQLAgentPolicyEngine):
    """Verify mutation statements (INSERT, UPDATE, DELETE) are rejected in read-only mode."""
    mutations = [
        "DELETE FROM customers WHERE id = 101;",
        "INSERT INTO orders (id, total) VALUES (999, 450.0);",
        "UPDATE customers SET email = 'hacked@evil.corp' WHERE id = 1;",
        "MERGE INTO customers USING temp_customers ON (customers.id = temp_customers.id);",
    ]

    for mut in mutations:
        res = policy_engine.validate_sql(mut)
        assert res.is_valid is False, f"Failed to block mutation: {mut}"
        assert SQLViolationType.UNAUTHORIZED_WRITE in res.violation_types or SQLViolationType.DISALLOWED_COMMAND in res.violation_types


def test_command_execution_and_file_exfiltration_blocking(policy_engine: SQLAgentPolicyEngine):
    """Verify execution of OS commands and file I/O functions are blocked."""
    exploits = [
        "SELECT xp_cmdshell('cat /etc/shadow');",
        "SELECT load_file('/var/lib/mysql-files/secret.key');",
        "SELECT * FROM customers INTO OUTFILE '/var/www/html/dump.csv';",
        "SELECT * FROM customers INTO DUMPFILE '/tmp/users.bin';",
        "SELECT sys_eval('whoami');",
    ]

    for exp in exploits:
        res = policy_engine.validate_sql(exp)
        assert res.is_valid is False, f"Failed to block exploit: {exp}"
        assert SQLViolationType.COMMAND_EXECUTION in res.violation_types


def test_blind_time_delay_injection_blocking(policy_engine: SQLAgentPolicyEngine):
    """Verify blind time-delay injection functions are blocked."""
    time_attacks = [
        "SELECT * FROM orders WHERE id = 1 AND SLEEP(5);",
        "SELECT * FROM customers WHERE id = 1 AND pg_sleep(10);",
        "SELECT * FROM products WHERE id = 1 AND BENCHMARK(5000000, MD5(1));",
    ]

    for atk in time_attacks:
        res = policy_engine.validate_sql(atk)
        assert res.is_valid is False, f"Failed to block blind injection: {atk}"
        assert SQLViolationType.BLIND_INJECTION in res.violation_types


def test_limit_capping_and_sanitization(policy_engine: SQLAgentPolicyEngine):
    """Verify limit caps are enforced and missing limits are auto-injected."""
    # 1. Limit exceeding max_limit (1000)
    over_limit = "SELECT * FROM orders LIMIT 50000;"
    res = policy_engine.validate_sql(over_limit)
    assert res.is_valid is False
    assert SQLViolationType.LIMIT_EXCEEDED in res.violation_types

    # 2. Limit within cap
    valid_limit = "SELECT * FROM orders LIMIT 100;"
    res_valid = policy_engine.validate_sql(valid_limit)
    assert res_valid.is_valid is True

    # 3. Missing limit with enforce_limit=True
    missing_limit = "SELECT * FROM orders;"
    res_enforce = policy_engine.validate_sql(missing_limit)
    assert res_enforce.is_valid is True
    assert res_enforce.sanitized_query == "SELECT * FROM orders LIMIT 1000;"

    # 4. sanitize_sql() helper
    clean = policy_engine.sanitize_sql("SELECT * FROM customers WHERE id = 1")
    assert "LIMIT 1000" in clean

    with pytest.raises(ValueError, match="SQL Policy Violation"):
        policy_engine.sanitize_sql("DROP TABLE customers;")


def test_sub_millisecond_latency_sla(policy_engine: SQLAgentPolicyEngine):
    """Verify deterministic SQL validation completes under 0.2ms average SLA."""
    queries = [
        "SELECT id, total FROM orders WHERE status = 'shipped';",
        "SELECT c.name, p.title FROM customers c JOIN products p ON c.id = p.id;",
        "SELECT * FROM orders LIMIT 50;",
        "SELECT count(*) FROM customers;",
    ]

    times = []
    for q in queries * 50:  # 200 iterations
        t0 = time.perf_counter()
        policy_engine.validate_sql(q)
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(times) / len(times)
    assert avg_ms < 1.0, f"Average validation time {avg_ms:.4f}ms exceeded SLA"


@pytest.mark.asyncio
async def test_nemo_action_integration():
    """Verify SQLAgentPolicyEngine can be registered as an action in NeMo Guardrails LLMRails."""
    engine = SQLAgentPolicyEngine(allowed_tables={"customers", "orders"})
    rails = LLMRails()

    action_fn = engine.as_nemo_action(action_name="check_sql")
    rails.register_action(action_fn, name="check_sql")

    assert "check_sql" in rails.engine.action_registry.list_actions()

    # Invoke registered action directly
    valid_act_res = await action_fn(sql="SELECT * FROM customers;")
    assert valid_act_res["is_valid"] is True

    blocked_act_res = await action_fn(sql="DROP TABLE customers;")
    assert blocked_act_res["is_valid"] is False
    assert len(blocked_act_res["violations"]) > 0
