# Design System — Scandoc 9000

## Overview
A document scanner that channels the warmth and tactility of early 90s office equipment — Xerox copiers, HP LaserJet control panels, System 7 Macs. Retro-nostalgic without being kitschy. The feeling of pressing a satisfying green START button on a warm beige machine. Clean, simple, a little bit fun.

**Design philosophy**: This is a copier, not a SaaS dashboard. Every element should feel like it belongs on a physical machine. Status readouts, not progress spinners. Chunky buttons, not floating action buttons. Paper, not pixels.

**Reference points**: Xerox 5100 copier interface, Apple LaserWriter control panel, early HP ScanJet software, Solarized Light palette, the green phosphor glow of a copier status display.

## Colors

### Primary Palette (from Stitch mockups — Material Design 3 tokens)
- **Surface** (`#fff9e9`): Page background — warm paper
- **Surface Container** (`#f4eedb`): Secondary surfaces
- **Surface Container High** (`#eee8d5`): Card backgrounds, settings panels
- **Surface Container Highest** (`#e8e2cf`): Emphasized surfaces
- **On-Surface** (`#1e1c10`): Primary text — deep ink
- **On-Surface Variant** (`#3d4947`): Secondary text
- **Outline** (`#6d7a78`): Borders, tertiary text
- **Outline Variant** (`#bcc9c7`): Subtle borders
- **Primary** (`#006a64`): Deep teal — for text accents, links
- **Primary Container** (`#2aa198`): Xerox green — START button, active nav, badges
- **Error** (`#ba1a1a`): STOP button, destructive actions
- **Tertiary** (`#aa3600`): Orange accent — warnings, secondary actions

### Functional Colors
- **Scanner Light** (`#859900`): Scanner lamp animation — bright green-yellow with glow
- **LCD Background** (`#002b36`): Dark status displays — the copier's LCD panel
- **LCD Text** (`#2aa198`): Teal text on LCD panels with `text-shadow: 0 0 8px rgba(42,161,152,0.6)`

### Full Tailwind Config
See `mockups/tailwind-config.js` for the complete token set used across all Stitch mockups.

## Typography

- **Display / Headers**: `'IBM Plex Mono'`, semibold — the copier control panel feel. Uppercase for major labels.
- **Body**: `'IBM Plex Sans'`, regular, 14-16px — readable, clean, slightly technical.
- **Status / Readouts**: `'IBM Plex Mono'`, regular, 13px — page counts, costs, processing status. Always monospace. The LCD display.
- **Labels**: `'IBM Plex Sans'`, medium, 11-12px, uppercase, letter-spacing 0.08em — section dividers, category tags.

**Fallback stack**: `-apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif`

**Load from Google Fonts**: `IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600`

## Spacing & Layout

- **Base unit**: 8px grid
- **Page padding**: 16px mobile, 24px desktop
- **Card padding**: 12-16px
- **Gap between elements**: 8-12px
- **Border radius**: 4px (subtle, not rounded — machines have edges)
- **Max content width**: 720px (centered on desktop)

## Components

### Buttons
- **Primary (Start)**: `Xerox Green` background, `Paper` text, 4px radius, bold uppercase label, 14px padding vertical, slight box-shadow (`0 2px 0 #1a8a82`) for a tactile "raised button" feel. On press: shadow collapses, translates down 2px.
- **Secondary**: `Paper Dark` background, `Ink` text, 1px solid `#c0b89d` border. Same press effect.
- **Danger**: `Red` background, white text. Rare — only for destructive actions.
- **Sizing**: Generous touch targets — minimum 44px height on mobile.

### Status Display (LCD Panel)
- Dark panel (`LCD Background`) with monospace text (`LCD Text`)
- Used for: processing log, current operation status, cost readout
- Subtle 1px inset border to feel recessed into the surface
- Optional scanline effect (very subtle repeating gradient, 2px)

### Document Cards
- `Paper Dark` background, 1px solid `#d6ceb5` border
- Title in `Ink`, semibold
- Metadata line in `Ink Light`, monospace for dates/costs
- Category badge: small pill, `Xerox Green` background on dark, or outlined on light
- Status indicators: green dot (filed), amber dot (processing), gray dot (queued)

### Upload Zone
- Dashed 2px border, `#c0b89d` color
- "Place document on glass" or "DROP PDF HERE" in uppercase monospace
- On dragover: border goes solid `Xerox Green`, subtle green glow

### Camera Viewfinder
- Black background with subtle rounded corners (the scanner glass)
- Thin green border when scanning is active (the scanner light)
- Motion indicator: monospace text overlay, top-right, on dark translucent background
- Flash effect on capture: brief white overlay (the copier flash)

### Progress / Scan Animation
- A horizontal green line that sweeps left-to-right across the element — mimicking the actual scanner light bar moving under the glass
- Used during: document processing, OCR in progress
- CSS animation: thin (3px) bright `Scanner Light` line, moves across width in ~2s, repeats

### Filter Pills (Library)
- Outlined by default: 1px border, `Ink Light` text
- Active: `Xerox Green` fill, `Paper` text
- Rounded but not fully round — 4px radius to match system

### Settings Panel
- Slide-out or collapsible, triggered by a gear icon or "SETTINGS" button
- Controls styled as physical switches/sliders where possible
- Grouped into sections with uppercase monospace headers

## Iconography
- Minimal. Prefer text labels over icons.
- Where icons are needed: simple line icons, 1.5px stroke, `Ink Light` color
- The rotate button can use ↻ as text — no icon library needed

## Motion & Animation
- **Scanner sweep**: Primary loading animation. Green line sweeps horizontally.
- **Button press**: 2px downward translate + shadow collapse. 100ms.
- **Flash on capture**: White overlay, 0 → 0.6 opacity instant, fade out 150ms.
- **Card entrance**: Subtle fade-in (200ms) when new documents appear in list.
- **No bouncing, no elastic, no parallax**. This is a machine. It moves with purpose.

## Do's and Don'ts

### Do
- Use monospace for any numerical readout (pages, cost, time, dates)
- Keep the warm paper background consistent — it's the signature
- Make buttons feel physical (shadow, press state)
- Show machine-like status messages ("PROCESSING 2 OF 5", "CLASSIFICATION COMPLETE")
- Use Xerox Green sparingly — it should feel like the one important button on the machine
- Maintain generous whitespace — copier UIs are not dense

### Don't
- Use cold whites or blue-tinted grays — everything is warm
- Use gradients (machines don't have gradients)
- Round corners more than 4px (this isn't a modern SaaS app)
- Animate with spring/bounce physics (machines move linearly)
- Use emoji in the UI (the copier doesn't have emoji)
- Put text over images without a solid background (readability first)
- Mix the dark LCD panel style with the light paper style in the same element

## Responsive Behavior
- **Mobile-first**: Single column, camera viewfinder takes full width
- **Desktop (>768px)**: Optional sidebar for library/files, main content stays centered at max 720px
- **The camera/scanner area is always the hero** — largest element on screen

## Dark Mode
- Not planned for v1. The warm paper aesthetic IS the brand. A dark mode would be a different app.
- Exception: the LCD status panel is always dark (it's a screen embedded in the machine)
