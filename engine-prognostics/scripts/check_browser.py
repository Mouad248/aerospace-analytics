import csv
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "reports/latest"
    data = json.loads((output / "results.json").read_text(encoding="utf-8"))
    private = ROOT / ".private/browser"
    private.mkdir(parents=True, exist_ok=True)
    errors, requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                requests.append(request.url) if request.url.startswith("http") else None
            ),
        )
        page.goto((output / "index.html").as_uri())
        exported_count = 0
        for name, record in data["subsets"].items():
            page.select_option("#subset", name)
            assert (
                int(page.locator("#cards .card strong").first.inner_text())
                == record["metrics"]["test_engines"]
            )
            page.select_option("#trajectory-select", "2")
            assert page.locator("#trajectory path").count() == 3
            page.locator("#horizon").evaluate(
                '(el)=>{el.value="80";el.dispatchEvent(new Event("input",{bubbles:true}));}'
            )
            assert page.locator("#horizon-value").inner_text() == "80 cycles"
            assert (
                page.locator("#engines tr").count() == record["metrics"]["test_engines"]
            )
            with page.expect_download() as pending:
                page.click("#export")
            path = private / f"{name}.csv"
            pending.value.save_as(str(path))
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == len(record["predictions"])
            for row, expected in zip(rows, record["predictions"], strict=True):
                assert (
                    int(row["engine"]) == expected["engine"]
                    and float(row["predicted_rul"]) == expected["rul_predicted"]
                )
                assert row["review_flag"] == str(expected["lower"] <= 80).lower()
            link = page.locator("#model-link").get_attribute("href")
            assert (output / link).resolve().is_file()
            exported_count += len(rows)
        page.select_option("#subset", "FD001")
        page.screenshot(path=str(private / "desktop.png"), full_page=True)
        assert not page.evaluate("document.documentElement.scrollWidth>innerWidth")
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth>innerWidth")
        page.screenshot(path=str(private / "mobile.png"), full_page=True)
        browser.close()
    assert not errors and not requests, (errors, requests)
    result = {
        "status": "passed",
        "subsets_checked": 4,
        "trajectory_controls_checked": True,
        "review_horizon_controls_checked": True,
        "downloaded_prediction_rows_verified": exported_count,
        "desktop": "1440x1050",
        "mobile": "390x844",
        "document_overflow": False,
        "javascript_errors": errors,
        "external_requests": requests,
    }
    (output / "browser_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
