#!/usr/bin/env python3
"""
Browser Bridge - Local WebSocket server that runs Playwright on YOUR machine.
Run this script on your computer to allow the cloud AI to control your local browser.

Usage:
  1. Install: pip install playwright websockets
  2. Install browsers: playwright install chromium
  3. Run: python browser_bridge.py
  4. In the app, set browser_bridge_url to ws://localhost:8765 (or ws://YOUR_IP:8765 if backend is remote)

The AI model runs in the cloud but sends commands to this bridge; Playwright runs locally and controls YOUR visible Chrome.
"""
import asyncio
import json
import logging
import sys

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Install playwright: pip install playwright && playwright install chromium")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = "0.0.0.0"  # Listen on all interfaces so remote backend can connect
PORT = 8765
HEADLESS = False  # Visible browser so you can see what the AI is doing
ACTION_TIMEOUT_MS = 15000  # Timeout for element-based actions (fill, click, etc.) so errors return sooner

# Global browser state
playwright = None
browser = None
context = None
page = None


async def ensure_browser():
    """Lazy-init Playwright browser on first command."""
    global playwright, browser, context, page
    if page is not None:
        return
    logger.info("Launching local browser (visible window)...")
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=HEADLESS, slow_mo=300)
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    )
    page = await context.new_page()
    logger.info("Local browser ready.")


async def handle_command(data: dict) -> dict:
    """Execute a single browser command and return result."""
    global playwright, browser, context, page
    action = data.get("action")
    if not action:
        return {"success": False, "error": "Missing 'action' in command"}

    await ensure_browser()

    try:
        if action == "navigate":
            url = data.get("url", "")
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            await page.goto(url, wait_until="networkidle", timeout=30000)
            return {"success": True, "result": f"Navigated to {url}"}

        elif action == "click":
            selector = data.get("selector", "")
            try:
                await page.click(selector, timeout=5000)
                return {"success": True, "result": f"Clicked {selector}"}
            except Exception:
                try:
                    await page.click(f"text={selector}", timeout=5000)
                    return {"success": True, "result": f"Clicked text={selector}"}
                except Exception:
                    await page.click(f"text=/{selector}/i", timeout=5000)
                    return {"success": True, "result": f"Clicked text matching {selector}"}

        elif action == "type":
            selector = data.get("selector", "")
            text = data.get("text", "")
            await page.fill(selector, text, timeout=ACTION_TIMEOUT_MS)
            return {"success": True, "result": f"Typed into {selector}"}

        elif action == "get_text":
            selector = data.get("selector", "")
            text = await page.text_content(selector, timeout=ACTION_TIMEOUT_MS)
            return {"success": True, "result": text or f"No text for {selector}"}

        elif action == "get_page_content":
            mode = data.get("mode", "")
            if mode == "summary":
                title = await page.title()
                url = page.url
                headings = await page.evaluate(
                    "() => Array.from(document.querySelectorAll('h1, h2, h3')).map(h => h.textContent.trim()).slice(0, 10)"
                )
                result = f"Page Title: {title}\nURL: {url}\nHeadings: {', '.join(headings)}"
            else:
                text = await page.evaluate("() => document.body.innerText")
                result = text[:5000] + ("..." if len(text) > 5000 else "")
            return {"success": True, "result": result}

        elif action == "screenshot":
            path = data.get("path", "screenshot.png")
            await page.screenshot(path=path)
            return {"success": True, "result": f"Screenshot saved to {path}"}

        elif action == "wait":
            value = data.get("value", "1")
            try:
                secs = float(value)
                await asyncio.sleep(secs)
                return {"success": True, "result": f"Waited {secs}s"}
            except ValueError:
                await page.wait_for_selector(value, timeout=10000)
                return {"success": True, "result": f"Waited for {value}"}

        elif action == "scroll":
            direction = (data.get("direction") or "").lower()
            if direction == "down":
                await page.evaluate("window.scrollBy(0, 500)")
            elif direction == "up":
                await page.evaluate("window.scrollBy(0, -500)")
            elif direction == "top":
                await page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                try:
                    px = int(direction)
                    await page.evaluate(f"window.scrollBy(0, {px})")
                except ValueError:
                    return {"success": False, "error": f"Invalid scroll: {direction}"}
            return {"success": True, "result": f"Scrolled {direction}"}

        elif action == "select_option":
            selector = data.get("selector", "")
            value = data.get("value", "")
            await page.select_option(selector, value, timeout=ACTION_TIMEOUT_MS)
            return {"success": True, "result": f"Selected {value} in {selector}"}

        elif action == "get_url":
            return {"success": True, "result": page.url}

        elif action == "close":
            if page:
                await page.close()
            if context:
                await context.close()
            if browser:
                await browser.close()
            if playwright:
                await playwright.stop()
            page = None
            context = None
            browser = None
            playwright = None
            return {"success": True, "result": "Browser closed"}

        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    except Exception as e:
        logger.exception("Command failed")
        return {"success": False, "error": str(e)}


async def handler(websocket):
    """Handle one WebSocket connection (one automation session). Keep connection open on errors."""
    logger.info("Client connected")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                response = await handle_command(data)
                await websocket.send(json.dumps(response))
            except json.JSONDecodeError as e:
                try:
                    await websocket.send(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
                except Exception:
                    raise
            except Exception as e:
                logger.exception("Error handling message, sending error and keeping connection open")
                try:
                    await websocket.send(json.dumps({"success": False, "error": str(e)}))
                except Exception:
                    raise
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        logger.info("Client disconnected")


async def main():
    async with websockets.serve(handler, HOST, PORT, ping_interval=20, ping_timeout=20):
        logger.info(f"Browser Bridge listening on ws://{HOST}:{PORT}")
        logger.info("Start browser automation from the app with browser_bridge_url = ws://YOUR_IP:{PORT}".format(PORT=PORT))
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
