"""
Shared constants for HTML-based evaluation metrics.
"""

# Viewport configurations for responsive testing
VIEWPORTS = {
    "mobile": {"width": 375, "height": 0},  # Mobile viewport (iPhone standard)
    "tablet": {"width": 1024, "height": 0},  # Tablet viewport (iPad standard)
    "desktop": {"width": 1280, "height": 0},  # Desktop viewport (standard HD)
}

# Minimum acceptable font size in pixels
MIN_FONT_SIZE = 12

# Minimum tap target size in pixels (48x48 for accessibility)
MIN_TAP_TARGET_SIZE = 48

# Minimum line-height to font-size ratio for readability
MIN_LINE_HEIGHT_RATIO = 1.5
