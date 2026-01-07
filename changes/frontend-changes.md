# Frontend Changes: Theme Toggle Button

## Overview

Added a dark/light theme toggle button to the Course Materials Assistant interface. The button is positioned in the top-right corner and allows users to switch between dark and light themes.

## Files Modified

### 1. `frontend/index.html`

Added the theme toggle button element with sun and moon SVG icons:

```html
<button id="themeToggle" class="theme-toggle" aria-label="Toggle dark/light theme" title="Toggle theme">
    <svg class="sun-icon">...</svg>
    <svg class="moon-icon">...</svg>
</button>
```

**Location:** Placed at the start of `<body>`, outside the container, to ensure fixed positioning works correctly.

### 2. `frontend/style.css`

Added the following CSS:

- **Theme Toggle Button Styles** (lines 8-71):
  - Fixed positioning in top-right corner
  - Circular button design (44x44px) matching existing aesthetic
  - Hover effects with scale and shadow
  - Focus ring for accessibility
  - Active state with scale-down effect
  - Sun/moon icon transition animations (rotate + scale + opacity)

- **Light Theme Variables** (lines 93-110):
  - Complete set of CSS custom properties for light mode
  - Uses `[data-theme="light"]` selector
  - Maintains primary blue accent color for consistency

- **Theme Transition Styles** (lines 112-128):
  - Smooth 0.3s transitions on all theme-affected elements
  - Prevents jarring color changes when toggling

- **Code Background Variable** (line 90 and 109):
  - Added `--code-bg` variable for code block backgrounds
  - Dark: `rgba(0, 0, 0, 0.2)`, Light: `rgba(0, 0, 0, 0.05)`

### 3. `frontend/script.js`

Added theme management functionality:

- **`initializeTheme()`**:
  - Checks localStorage for saved preference
  - Falls back to system preference via `prefers-color-scheme`
  - Listens for system theme changes

- **`setTheme(theme)`**:
  - Applies theme by setting/removing `data-theme` attribute on `<html>`
  - Updates ARIA label dynamically for screen readers

- **`toggleTheme()`**:
  - Toggles between themes
  - Persists choice to localStorage

- **Event Listeners**:
  - Click handler for mouse interaction
  - Keyboard handler for Enter and Space keys

## Features

### Design
- Circular button with existing border and surface colors
- Smooth hover animation (scale up + glow)
- Icon transition: rotating scale animation when switching themes
- Consistent with existing button patterns (focus ring, hover states)

### Accessibility
- `aria-label` attribute with current/target theme information
- `title` attribute for tooltip
- Keyboard navigation support (Tab to focus, Enter/Space to activate)
- Focus ring visible on keyboard navigation
- Dynamic ARIA label updates on theme change

### User Experience
- Theme preference persisted to localStorage
- Respects system preference when no saved preference exists
- Listens for system preference changes (e.g., OS-level dark mode toggle)
- Smooth 0.3s transitions prevent jarring color changes

## Light Theme Color Palette

| Variable | Dark Value | Light Value |
|----------|------------|-------------|
| `--background` | `#0f172a` | `#f8fafc` |
| `--surface` | `#1e293b` | `#ffffff` |
| `--surface-hover` | `#334155` | `#f1f5f9` |
| `--text-primary` | `#f1f5f9` | `#1e293b` |
| `--text-secondary` | `#94a3b8` | `#64748b` |
| `--border-color` | `#334155` | `#e2e8f0` |
| `--assistant-message` | `#374151` | `#f1f5f9` |
| `--shadow` | `rgba(0,0,0,0.3)` | `rgba(0,0,0,0.1)` |

## Testing

To test the toggle button:

1. Start the server: `./run.sh`
2. Open http://localhost:8000
3. Click the theme toggle button in the top-right corner
4. Verify smooth transition between themes
5. Refresh the page - theme preference should persist
6. Test keyboard navigation (Tab to button, Enter to toggle)
7. Test with system dark mode preference changes
