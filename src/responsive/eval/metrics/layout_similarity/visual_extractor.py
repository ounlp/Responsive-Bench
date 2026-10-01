"""
Visual component extractor for layout similarity computation.

Extracts visual components (images, text blocks, buttons, etc.) from HTML pages
using Playwright and returns normalized element data with bounding boxes.
"""

import io
import os
import traceback
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

from .box_utils import boxes_adjacent, merge_boxes, is_within


def extract_visual_components(url, save_path, extra_out=None, viewport=None):
    """
    Args:
        viewport: Optional {"width": int, "height": int}. When omitted, Playwright's
            own default (~1280x720) is used, preserving prior behavior for callers
            that don't care about a specific viewport (e.g. dev/annotation tools).
            Explicitly passing viewport=None to Playwright's new_page() instead
            disables fixed-viewport emulation entirely, which is why this only
            forwards the kwarg when a viewport dict is actually given.
    """
    # Convert local path to file:// URL if it's a file
    if isinstance(url, Path):
        url = str(url)
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    screenshot_image = None
    element_data = {}

    try:
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(args=["--allow-file-access-from-files"])
            page = browser.new_page(viewport=viewport) if viewport is not None else browser.new_page()

            # Navigate to the URL and wait for network to settle
            page.goto(url, wait_until="networkidle", timeout=60000)

            total_height = page.evaluate("() => document.documentElement.scrollHeight")
            viewport_height = page.evaluate("() => window.innerHeight")

            # Scroll through the full page to trigger lazy-loaded images/content
            scroll_pos = 0
            while scroll_pos < total_height:
                page.evaluate(f"window.scrollTo(0, {scroll_pos})")
                page.wait_for_timeout(80)
                scroll_pos += viewport_height
            page.evaluate("window.scrollTo(0, 0)")

            # Wait for any images that started loading during scroll to finish.
            # Bounded by a race against a timeout: headless Chromium sometimes
            # never fires load/error for a lazy-loaded <img> (observed with
            # pages that repeat the same src across many <img loading="lazy">
            # elements), which would otherwise hang this evaluate() forever --
            # unlike page.goto/page.screenshot below, evaluate() has no
            # built-in timeout of its own.
            page.evaluate("""() => Promise.race([
                Promise.all(
                    Array.from(document.images)
                        .filter(img => !img.complete)
                        .map(img => new Promise(resolve => {
                            img.addEventListener('load', resolve);
                            img.addEventListener('error', resolve);
                        }))
                ),
                new Promise(resolve => setTimeout(resolve, 5000))
            ])""")

            total_width = page.evaluate("() => document.documentElement.scrollWidth")
            total_height = page.evaluate("() => document.documentElement.scrollHeight")

            selectors = {
                'video': 'video',
                'image': 'img',
                'text_block': 'p, span, a, strong, h1, h2, h3, h4, h5, h6, li, th, td, label, code, pre, div',
                'form_table': 'form, table, div.form',
                'button': 'button, input[type="button"], input[type="submit"], [role="button"]',
                'nav_bar': 'nav, [role="navigation"], .navbar, [class~="nav"], [class~="navigation"], [class~="menu"], [class~="navbar"], [id="menu"], [id="nav"], [id="navigation"], [id="navbar"]',
                'divider': 'hr, [class*="separator"], [class*="divider"], [id="separator"], [id="divider"], [role="separator"]',
            }

            for key, value in selectors.items():
                element_data[key] = []
                elements = page.query_selector_all(value)
                for element in elements:
                    if not element.is_visible():
                        continue

                    box = element.bounding_box()
                    if box and box['width'] > 0 and box['height'] > 0:  # Only add if the element has a visible bounding box
                        text_content = None
                        if key == 'text_block':
                            is_direct_text = element.evaluate("""
                                (el) => Array.from(el.childNodes).some(node =>
                                    node.nodeType === Node.TEXT_NODE && node.textContent.trim() !== '')
                            """)
                            tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                            if tag_name == 'div' and not is_direct_text:  # Skip div elements that do not contain direct text
                                continue

                            text_content = element.text_content().strip()
                            if not text_content:  # Skip empty text blocks
                                continue

                        element_data[key].append({
                            'type': key,
                            'box': {'x': box['x'], 'y': box['y'], 'width': box['width'], 'height': box['height']},
                            'text_content': text_content,
                        })

            # Process adjacent text blocks
            text_blocks = element_data['text_block']
            text_blocks.sort(key=lambda block: (block['box']['y'], block['box']['x']))
            merged_text_blocks = []
            while text_blocks:
                current = text_blocks.pop(0)
                index = 0
                add = True
                while index < len(text_blocks):
                    if is_within(text_blocks[index]['box'], current['box']):
                        # Skip nested text blocks
                        del text_blocks[index]
                        continue
                    elif is_within(current['box'], text_blocks[index]['box']):
                        # the current box is nested within another box, skip the current box
                        add = False
                        break

                    if boxes_adjacent(current['box'], text_blocks[index]['box']):
                        if current['box']['x'] < text_blocks[index]['box']['x'] or current['box']['y'] < text_blocks[index]['box']['y']:
                            current['text_content'] += " " + text_blocks[index]['text_content']
                        else:
                            current['text_content'] = text_blocks[index]['text_content'] + " " + current['text_content']

                        current['box'] = merge_boxes(current['box'], text_blocks[index]['box'])
                        del text_blocks[index]
                    else:
                        index += 1

                if add:
                    merged_text_blocks.append(current)

            # Combine merged text blocks with other elements
            element_data['text_block'] = merged_text_blocks

            # Take the full page screenshot
            image_bytes = page.screenshot(full_page=True, animations="disabled", timeout=60000)

            # Convert bytes to a PIL Image
            image_buffer = io.BytesIO(image_bytes)
            screenshot_image = Image.open(image_buffer)

            # Optional side-channel for callers that need the pre-annotation screenshot
            # (e.g. to redraw boxes after the metric's own post-hoc text_block merge).
            if extra_out is not None:
                extra_out['clean_image'] = screenshot_image.copy()
                extra_out['total_width'] = total_width
                extra_out['total_height'] = total_height

            # Draw bounding boxes and labels
            draw = ImageDraw.Draw(screenshot_image)
            font = ImageFont.load_default()  # Using default font

            for key in element_data:
                for item in element_data[key]:
                    x = item['box']['x']
                    y = item['box']['y']
                    width = item['box']['width']
                    height = item['box']['height']
                    draw.rectangle(((x, y), (x + width, y + height)), outline="red", width=2)
                    draw.text((x, y), item['type'], fill="red", font=font)

            # Save the annotated image to a file
            if save_path:
                screenshot_image.save(save_path)
                print(f"Saved annotated image to {save_path}")

            # finally, normalize the positions and sizes of each element to relative values before returning
            for key in element_data:
                for item in element_data[key]:
                    box = item['box']
                    item['box'] = {
                        'x': box['x'] / total_width,
                        'y': box['y'] / total_height,
                        'width': box['width'] / total_width,
                        'height': box['height'] / total_height
                    }

            browser.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        print(traceback.format_exc())
        screenshot_image = Image.new('RGB', (1280, 960), color='white')
        if save_path:
            screenshot_image.save(save_path)

    return element_data
