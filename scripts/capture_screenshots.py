from playwright.sync_api import sync_playwright
from pathlib import Path

IMG_DIR = Path("docs/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1366, "height": 850})
    page.goto("http://localhost:8501", wait_until="networkidle")
    
    # 1. Main View (Health Status)
    page.get_by_role("tabpanel", name="Health Status").get_by_text(
        "Battery Health Status"
    ).wait_for(state="visible")
    page.screenshot(path=str(IMG_DIR / "dashboard_view_health_status.png"))
    print("Captured Health Status")

    # Click Tab 2: SoH Trend
    page.get_by_role("tab", name="SoH Trend").click()
    trend_panel = page.get_by_role("tabpanel", name="SoH Trend")
    trend_panel.locator('[data-testid="stVegaLiteChart"]').last.wait_for(
        state="visible"
    )
    page.screenshot(path=str(IMG_DIR / "dashboard_view_soh_trend.png"))
    print("Captured SoH Trend")

    # Click Tab 3: SoH Predictions & EOL
    page.get_by_role("tab", name="SoH Predictions & EOL").click()
    predictions_panel = page.get_by_role("tabpanel", name="SoH Predictions & EOL")
    predictions_panel.locator('[data-testid="stVegaLiteChart"]').last.wait_for(
        state="visible"
    )
    predictions_panel.get_by_text("Observed EOL Cycle").wait_for(state="visible")
    page.screenshot(path=str(IMG_DIR / "dashboard_view_predictions_rul.png"))
    print("Captured Predictions & RUL")

    # Battery B0006 warning state
    page.get_by_role("tab", name="Health Status").click()
    page.get_by_role("combobox", name="Select Battery").click()
    page.get_by_role("option", name="B0006").click()
    page.get_by_text("Battery in POOR condition").wait_for(state="visible")
    page.screenshot(path=str(IMG_DIR / "dashboard_view_warning_state.png"))
    print("Captured Warning State")

    browser.close()
    print("All captures completed successfully!")
