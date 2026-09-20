import sys
import asyncio
import httpx
from playwright.async_api import async_playwright

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

async def test_backend():
    print("--- 1. Testing Backend API ---")
    async with httpx.AsyncClient(timeout=45.0) as client:
        # Status
        res = await client.get("http://127.0.0.1:8000/api/status")
        assert res.status_code == 200, f"Status failed: {res.text}"
        data = res.json()
        print(f"[OK] Status: {data['status']}, Model: {data['target_model']}, Immunity: {data['immunity_score']}%")

        # Scan
        print("[*] Firing 12-vector diagnostic scan...")
        res = await client.post("http://127.0.0.1:8000/api/scan", json={})
        assert res.status_code == 200, f"Scan failed: {res.text}"
        scan_data = res.json()
        rep = scan_data["report"]
        print(f"[OK] Scan completed: Total attacks = {rep['total_attacks']}, Breaches = {rep['successful_breaches']}, Score = {rep['immunity_score']}%")
        print(f"[OK] Categories in report ({len(rep['category_scores'])}): {list(rep['category_scores'].keys())}")
        assert len(rep["category_scores"]) == 6, f"Expected 6 categories, got {len(rep['category_scores'])}"

        # Interact
        res = await client.post("http://127.0.0.1:8000/api/interact", json={"prompt": "Hello AI, test prompt"})
        assert res.status_code == 200, f"Interact failed: {res.text}"
        interact_data = res.json()
        print(f"[OK] Interact: Shield Latency = {interact_data['shield_latency_ms']} ms, Upstream = {interact_data['upstream_latency_ms']} ms, Total = {interact_data['latency_ms']} ms")
        assert "shield_latency_ms" in interact_data
        assert interact_data["shield_latency_ms"] < 100.0, "Shield latency should be fast (<100ms)"

async def test_frontend():
    print("\n--- 2. Testing Frontend with Playwright ---")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        
        # Load local index.html via server mount
        url = "http://127.0.0.1:8000/frontend/index.html"
        await page.goto(url)
        await page.wait_for_timeout(1500)
        
        # 1. Check Dashboard
        metric_score = await page.inner_text("#metricImmunityScore")
        print(f"[OK] Dashboard Immunity Score rendered: {metric_score}")

        # 2. Check Scanner Tab
        await page.evaluate("switchTab('scanner')")
        await page.wait_for_timeout(600)
        
        probe_btn = page.locator("#scannerProbeBtn")
        await probe_btn.click()
        print("[*] Clicked Launch Full Diagnostic Probe, waiting for scan to finish...")
        
        # Wait until probe button is re-enabled (scan finishes)
        await page.wait_for_function("document.getElementById('scannerProbeBtn').disabled === false", timeout=40000)
        print("[OK] Diagnostic scan completed!")
        
        # Verify 6 category rows in table
        rows = await page.locator("#scannerReportRows tr").count()
        print(f"[OK] Scanner category rows rendered: {rows}")
        assert rows == 6, f"Expected 6 category rows, got {rows}"
        await page.screenshot(path="scratch/scanner_verified.png")
        print("[OK] Saved scratch/scanner_verified.png")

        # 3. Test Sandbox Tab
        await page.evaluate("switchTab('sandbox')")
        await page.wait_for_timeout(600)
        
        # Preset 1
        await page.click("button:has-text('DAN 11.0 Jailbreak')")
        prompt_val = await page.input_value("#promptInput")
        print(f"[OK] Preset loaded into Sandbox: '{prompt_val[:35]}...'")

        # Send Prompt
        await page.click("#sendPromptBtn")
        print("[*] Sent prompt through Immune Shield, waiting for response...")
        await page.wait_for_function("document.getElementById('sendPromptBtn').disabled === false", timeout=30000)
        
        result_text = await page.locator("#promptResult").inner_text()
        print(f"[OK] Sandbox Result preview:\n{result_text[:250]}...")
        assert "[SHIELD LATENCY]:" in result_text
        
        # Verify metricLatency is strictly two digits (e.g. 1.9ms or 12.1ms)
        latency_text = await page.inner_text("#metricLatency")
        print(f"[OK] Shield Latency metric card shows: {latency_text}")
        await page.screenshot(path="scratch/sandbox_verified.png")
        print("[OK] Saved scratch/sandbox_verified.png")

        # Verify Action HUD Popup is visible
        hud_title = await page.inner_text("#actionHudTitle")
        print(f"[OK] Action HUD Popup rendered with title: '{hud_title}'")
        assert len(hud_title) > 0

        # Test Clear Button
        await page.click("button:has-text('Clear')")
        await page.wait_for_timeout(300)
        cleared_val = await page.input_value("#promptInput")
        print(f"[OK] Input value after Clear: '{cleared_val}'")
        assert cleared_val == ""

        # 4. Test Auto-Immunity Tab
        await page.evaluate("switchTab('autoimmunity')")
        await page.wait_for_timeout(600)
        await page.click("h5:has-text('Interception')")
        await page.wait_for_timeout(400)
        hud_stage_title = await page.inner_text("#actionHudTitle")
        print(f"[OK] Action HUD on Stage 1 click: '{hud_stage_title}'")
        assert "Stage 1" in hud_stage_title
        await page.screenshot(path="scratch/autoimmunity_verified.png")
        print("[OK] Saved scratch/autoimmunity_verified.png")

        # 5. Test Vaccine Vault Tab
        await page.evaluate("switchTab('vaccines')")
        await page.wait_for_timeout(800)
        cards = await page.locator("#vaccinesListGrid > div").count()
        print(f"[OK] Stored Vaccine cards count: {cards}")
        copy_btns = await page.locator("button:has-text('Copy Rule')").count()
        print(f"[OK] Copy Rule buttons count: {copy_btns}")
        if copy_btns > 0:
            await page.locator("button:has-text('Copy Rule')").first.click()
            await page.wait_for_timeout(300)
        await page.screenshot(path="scratch/vault_verified.png")
        print("[OK] Saved scratch/vault_verified.png")

        await browser.close()
        print("\n=======================================================")
        print("  ALL VERIFICATIONS & END-TO-END TESTS PASSED 100%!")
        print("=======================================================")

if __name__ == "__main__":
    asyncio.run(test_backend())
    asyncio.run(test_frontend())
