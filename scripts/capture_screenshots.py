from playwright.sync_api import sync_playwright
import time
from pathlib import Path

IMG_DIR = Path("docs/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1366, "height": 850})
    page.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(2)
    
    # 1. Main View (Health Status)
    page.screenshot(path=str(IMG_DIR / "dashboard_view_health_status.png"))
    print("Captured Health Status")

    # Click Tab 2: SoH Trend
    page.click("text=SoH Trend")
    time.sleep(2)
    page.screenshot(path=str(IMG_DIR / "dashboard_view_soh_trend.png"))
    print("Captured SoH Trend")

    # Click Tab 3: Predictions
    page.click("text=Predictions")
    time.sleep(2)
    page.screenshot(path=str(IMG_DIR / "dashboard_view_predictions_rul.png"))
    print("Captured Predictions & RUL")

    # Battery B0018 Warning State
    page.click("text=Health Status")
    time.sleep(1)
    # Select B0018
    page.locator('div[data-baseweb="select"]').first.click()
    time.sleep(0.5)
    page.locator("li:has-text('B0018')").first.click()
    time.sleep(2)
    page.screenshot(path=str(IMG_DIR / "dashboard_view_warning_state.png"))
    print("Captured Warning State")

    browser.close()
    print("All captures completed successfully!")
