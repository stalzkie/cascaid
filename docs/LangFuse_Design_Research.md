# LangFuse UI Design System — Research

Primary-source investigation of LangFuse's actual shipped design tokens
(github.com/langfuse/langfuse, `main` branch, fetched 2026-08-29), done to
ground a rewrite of `frontend/src/index.css` and the corner radii in
`frontend/src/components/PipelineGraph.tsx`. Every claim below is sourced to
the exact file fetched from `raw.githubusercontent.com` or a fetched
component file at that path/URL — no secondary blog posts were used.

## Verdict up front

**LangFuse does not use sharp (near-zero) corners.** Its `web/` app is a
stock shadcn/ui (`style: "default"`, `baseColor: "slate"`) setup, and it ships
shadcn's **unmodified default radius**: `--radius: 0.5rem` (8px) in both
light and dark mode
([`web/src/styles/globals.css`](https://raw.githubusercontent.com/langfuse/langfuse/main/web/src/styles/globals.css),
lines 392 and 564). Buttons, inputs, and badges round to 6px
(`rounded-md`), cards to 8px (`rounded-lg`), the smallest control state to
4px (`rounded-sm`) — nothing rounds to 0. The one place LangFuse *is*
genuinely sharp is its **data tables**: rows and cells carry no radius at
all, just hairline `border-b` separators. If "match LangFuse exactly" means
literal fidelity, the honest answer is: LangFuse's edges are softly rounded
(4–8px), not sharp — the "sharp, professional, white" read people associate
with LangFuse comes from its flat white surfaces, hairline borders, and
restrained shadows, not from zero border-radius. See "Where this conflicts
with 'sharp edges'" below for the decision this leaves open.

What *is* true to the "sharp/professional" instinct: pure white background
in both page and card surfaces, borders (not shadows) doing almost all of
the elevation work, a near-black neutral palette instead of a loud brand
color, and a tight, small-type data-table density. Those are the parts worth
porting exactly.

## Sources fetched

| Source | URL |
|---|---|
| Global CSS / design tokens | `raw.githubusercontent.com/langfuse/langfuse/main/web/src/styles/globals.css` |
| shadcn config | `raw.githubusercontent.com/langfuse/langfuse/main/web/components.json` |
| Button primitive | `raw.githubusercontent.com/langfuse/langfuse/main/web/src/components/ui/button.tsx` |
| Card primitive | `raw.githubusercontent.com/langfuse/langfuse/main/web/src/components/ui/card.tsx` |
| Badge primitive | `raw.githubusercontent.com/langfuse/langfuse/main/web/src/components/ui/badge.tsx` |
| Table primitives | `raw.githubusercontent.com/langfuse/langfuse/main/web/src/components/ui/table.tsx` |
| Input primitive | `raw.githubusercontent.com/langfuse/langfuse/main/web/src/components/ui/input.tsx` |
| Status/level color mapping | `raw.githubusercontent.com/langfuse/langfuse/main/web/src/components/level-colors.tsx` |
| Trace/pipeline graph node renderer | `raw.githubusercontent.com/langfuse/langfuse/main/web/src/features/trace-graph-view/components/GraphNode.tsx` |
| Sidebar shell | `raw.githubusercontent.com/langfuse/langfuse/main/web/src/components/nav/AppSidebar/AppSidebar.tsx` |
| `package.json` (Tailwind/Next/React versions) | `raw.githubusercontent.com/langfuse/langfuse/main/web/package.json` |
| Tailwind CSS v4 `shadow-*` defaults (third party, used only to decode LangFuse's un-overridden `shadow-xs` utility) | `tailwindcss.com/docs/box-shadow` |

Stack context: Next.js 16.3.3, React 19.2.4, Tailwind CSS `^4.2.2`
(`web/package.json`). Tailwind v4 is CSS-first — there is **no**
`tailwind.config.ts` in `web/`; all theme tokens live directly in
`globals.css` via `@theme` blocks. `components.json` confirms
`"style": "default"`, `"baseColor": "slate"`, `"cssVariables": true` — this
is shadcn's stock "slate" preset, not a custom theme built from scratch.

## 1. Border radius

| Token | Value | Computed | Source |
|---|---|---|---|
| `--radius` (base) | `0.5rem` | 8px | `globals.css` L392 (`:root`), L564 (`.dark`) — identical in both modes |
| `--radius-sm` | `calc(var(--radius) - 4px)` | 4px | `globals.css` L147 |
| `--radius-md` | `calc(var(--radius) - 2px)` | 6px | `globals.css` L148 |
| `--radius-lg` | `var(--radius)` | 8px | `globals.css` L149 |

Per-component usage (all confirmed by reading the actual component source,
not assumed from the tokens):

| Component | Class used | Radius |
|---|---|---|
| `Button` (default/sm/lg) | `rounded-md` | 6px | `ui/button.tsx` |
| `Button` (`size="xs"`) | `rounded-sm` | 4px | `ui/button.tsx` |
| `Card` | `rounded-lg` | 8px | `ui/card.tsx` |
| `Badge` (all variants) | `rounded-md` | 6px | `ui/badge.tsx` |
| `Input` | `rounded-md` | 6px | `ui/input.tsx` |
| `Table` / `TableRow` / `TableCell` | *(none)* | 0px — square | `ui/table.tsx` |
| Trace-graph node box (`GraphNode`) | `rounded-md` | 6px | `trace-graph-view/components/GraphNode.tsx` |

Notable: **Badges are not pills.** LangFuse's `Badge` component uses
`rounded-md` (6px) unconditionally — there is no `rounded-full` badge
variant in `badge.tsx`. Status/score chips (`success`/`error`/`warning`
variants) get the same 6px radius as every other badge.

The trace-graph node component is the closest real analog to Cascaid's
`PipelineGraph.tsx` — it renders nodes as absolutely-positioned `<div>`s
(not SVG) with `rounded-md px-2 ... border-2`, i.e. 6px corners and a
**2px** border (thicker than the 1px hairline used almost everywhere else,
used here specifically so per-type border color reads clearly against the
white canvas).

## 2. Color palette

All neutral tokens are literally the values shipped by shadcn/ui's stock
**"slate"** base-color theme (recognizable HSL triplets:
`222.2 84% 4.9%` foreground, `210 40% 96.1%` muted, `214.3 31.8% 91.4%`
border, etc.) — LangFuse's `components.json` (`"baseColor": "slate"`)
confirms this was a deliberate starting point, and `globals.css` shows they
kept these specific values rather than overriding them, then layered ~40
custom extension tokens on top (status pairs, sidebar ladder, syntax-highlight
colors, chart colors).

### Light mode (`:root`, `globals.css` L268–435)

| Token | HSL (as shipped) | Hex (computed) | Role |
|---|---|---|---|
| `--background` | `0 0% 100%` | `#ffffff` | page canvas |
| `--card` | `0 0% 100%` | `#ffffff` | card surface — **identical to background**, elevation comes from border + a whisper of shadow, not a darker fill |
| `--foreground` | `222.2 84% 4.9%` | `#020817` | primary text — near-black navy, not pure black |
| `--muted` | `210 40% 96.1%` | `#f1f5f9` | subtle fill (hover rows, disabled bg) |
| `--muted-foreground` | `215.4 16.3% 46.9%` | `#64748b` | secondary text |
| `--foreground-tertiary` | `215 12% 62%` | `#929caa` | placeholders / disabled / hints (documented as "flagged in review, kept per design preference," ~2.8:1 contrast — a deliberate low-contrast exception, not an accident) |
| `--border` | `214.3 31.8% 91.4%` | `#e2e8f0` | default hairline |
| `--border-contrast` | `214.3 20% 80%` | — | stronger line for tree connectors / timeline grid, "one clear step above the hairline" |
| `--primary` | `222.2 47.4% 11.2%` | `#0f172a` | default button fill — **near-black, not the brand color** |
| `--primary-accent` | `243 75.4% 58.6%` | `#4e46e5` | the actual brand/indigo accent — used for tabs, selection state, focus, not default buttons |
| `--link` | `246 40% 50%` | `#574db3` | hyperlink color, deliberately muted vs. `--primary-accent` |
| `--destructive` | `0 84.2% 60.2%` | `#ef4444` | error/danger |
| `--header` | `210 40% 98%` | `#f8fafc` | app header bar |

Notably: `--primary-accent: #4e46e5` is within one digit of Cascaid's own
`--accent: #4f46e5` (`frontend/src/index.css` L22) — both landed on the same
indigo-600-adjacent hue independently.

### Status/semantic pairs (light, `globals.css` L354–379)

LangFuse encodes status as **light/dark pairs per hue** — a light tinted fill
(`--light-*`, alpha baked in) plus a readable foreground (`--dark-*`) —
which is exactly the shape of a badge-background/badge-text pair:

| Hue | `--light-*` (fill) | `--dark-*` (text) | Used for |
|---|---|---|---|
| red | `oklch(93.6% 0.032 17.717 / 0.7)` | `oklch(63.7% 0.237 25.331)` | `Badge variant="error"`, `ObservationLevel.ERROR` |
| yellow | `oklch(97.3% 0.071 103.193 / 0.7)` | `oklch(47.6% 0.114 61.907)` | `Badge variant="warning"`, `ObservationLevel.WARNING` |
| green | `oklch(95% 0.052 163.051 / 0.7)` | `oklch(59.6% 0.145 163.225)` | `Badge variant="success"`, default/ok level |
| blue | `214 94.6% 92.7%` (hsl) | `219 85% 52%` (hsl) | score-level tag: observation |
| violet | `251 91% 95.5%` (hsl) | `262 72% 54%` (hsl) | score-level tag: trace |
| teal | `167 85% 89%` (hsl) | `175 84% 28%` (hsl) | score-level tag: session |

Source: `globals.css` L354–379 plus `ui/badge.tsx` for the variant → token
mapping, plus `level-colors.tsx` for the trace-status → hue mapping
(`DEFAULT` → green status bar/no chip, `DEBUG` → neutral gray chip,
`WARNING` → yellow chip, `ERROR` → red chip;
`level-colors.tsx` L8–12, L35–41). This 6-hue "light fill / dark text" pair
system is functionally identical in shape to Cascaid's separate
`--status-*` ramp and `--node-*` identity set — see the mapping section
below.

### Dark mode (`.dark`, `globals.css` L437–593)

Dark mode is fully supported and is **not** just an inverted light palette —
the comment at L438–450 documents a deliberate Carbon-style "layers get one
step lighter" ladder:

| Token | HSL | Hex | Role |
|---|---|---|---|
| `--background` | `0 0% 6%` | `#0f0f0f` | canvas (darkest content tier) |
| `--card` | `0 0% 7.5%` | `#131313` | one step above canvas |
| `--modal` | `0 0% 9%` | — | one step above card |
| `--popover` | `0 0% 12%` | — | brightest surface tier (menus outrank dialogs) |
| `--foreground` | `0 0% 70%` | `#b3b3b3` | body text — **capped at 70%, never near-white** ("never near-white," comment L456) |
| `--border` | `0 0% 15%` | `#262626` | must read on every tier including popover |
| `--destructive` | `0 78.7% 63.1%` | `#eb5757` | |
| `--radius` | `0.5rem` | 8px | **unchanged from light mode** — radius does not shift with theme |

Sidebar chrome is treated as its own darkest tier (`--sidebar-background:
0 0% 2%` in dark vs. `0 0% 100%`/white in light, both from `globals.css`
L393 and L566), while in light mode the sidebar is plain white with only a
`sidebar-border` hairline — no distinct color block.

## 3. Typography

Source: `globals.css` L12–38 (font tokens) and L157–170 (size scale).

- **No custom webfont.** `--font-sans` and `--font-mono` are both **system
  stacks**: `ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", ...`
  and `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation
  Mono", "Courier New", monospace`. A block comment (L12–23) explicitly
  documents this as **deliberate**: "the typeface is deliberately DEFERRED:
  the app ships the system stacks while the token system ... stays fully
  font-relative," with a note that a future webfont swap would go through
  `next/font`.
- **Exactly two font weights exist app-wide**: `--font-weight-regular: 400`
  and `--font-weight-bold: 600` (L37–38). `--font-weight-bold` deliberately
  overrides Tailwind's built-in 700 — `font-bold` *is* the bold role, and
  `font-medium`/`font-semibold`/raw weight numbers are flagged as "drift" in
  the source comment.
- **Size scale** (all `rem`, each carrying its own weight token so a
  `text-*` class alone yields a complete style):

  | Token | rem | px | Weight |
  |---|---|---|---|
  | `--text-xs` | 0.7rem | 11.2px | regular |
  | `--text-sm` | 0.825rem | 13.2px | regular |
  | `--text-base` | 0.9rem | 14.4px | regular |
  | `--text-lg` | 1.1rem | 17.6px | regular |
  | `--text-xl` | 1.2rem | 19.2px | regular |
  | `--text-2xl` | 1.3rem | 20.8px | regular |
  | `--text-3xl` | 1.5rem | 24px | regular |

  Base body text is **14.4px, smaller than the browser default (16px)** —
  consistent with a dense data-tool UI, not a marketing page.

- **Numeric formatting**: `tabular-nums` is used explicitly throughout the
  app for numeric values — confirmed via a repo-wide code search (21 hits)
  including `GraphNode.tsx` (the per-node observation counter, e.g. "(2/3)"),
  cost-estimate rows, chart values, and the time-picker input. This is a
  direct analog for Cascaid's risk scores / run IDs / timestamps, which
  should carry the same treatment.
- Monospace usage in components is more limited than sans — `Table`/`Badge`
  don't force `font-mono`; it's applied ad hoc to numeric/code-like values.

## 4. Spacing / density (data tables)

Source: `ui/table.tsx`.

- `Table` root: `text-sm` (13.2px) base, `table-fixed`,
  `border-separate border-spacing-0` (so per-cell borders render crisply
  with no double-border artifacts).
- `TableHeader`/`TableHead`: header cells are `h-10` (40px tall),
  `font-bold`, `text-muted-foreground`, with a sticky `bg-background`
  and a `border-b` — no header background tint, just weight + color to
  differentiate from body rows.
- `TableBody`: explicitly downsizes to **`text-xs` (11.2px)** — body rows
  are visibly smaller than the header. `tr:last-child` drops its border.
- `TableCell`: has a `density` prop, `"compact"` (default, `px-2 py-0` —
  essentially zero vertical cell padding, height comes from line-height
  alone) vs. `"comfortable"` (`p-2`, 8px all around). Compact is the
  default for data-dense views.
- `TableRow`: `hover:bg-muted/50`, `data-[state=selected]:bg-muted`,
  `border-b` — row separation is a **1px bottom hairline**, never a
  shadow or alternating-stripe fill.
- No radius anywhere in the table primitives — square corners top to
  bottom, confirmed above.

## 5. Borders vs. shadows for elevation

Source: `ui/card.tsx`, `ui/button.tsx`, `ui/table.tsx`, `globals.css`
(no custom `--shadow-*` tokens are defined anywhere in the file, and a
repo-wide search for `box-shadow` under `web/src/styles` returns zero
results), decoded against Tailwind CSS v4's built-in shadow scale
(tailwindcss.com/docs/box-shadow, since LangFuse doesn't redefine it):

| Element | Elevation technique |
|---|---|
| `Card` | `border` **+** `shadow-xs` → `box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05)` — a near-imperceptible 1px/2px, 5%-opacity shadow, paired with a 1px border. Not flat-flat, but very close to it. |
| `Button` | border only (`outline` variant); filled variants (`default`, `secondary`) have **no shadow at all** |
| `Input` | border only, no shadow |
| `Badge` | border only (`border-transparent` for filled variants, so effectively none) |
| `Table` rows/cells | 1px `border-b` only, no shadow |
| Trace-graph node (`GraphNode`) | `border-2` (thicker, per-type colored) + `ring` on hover/select/active — no shadow for the resting state; a `shadow-[0_0_16px_...]` glow is used only for the "active/playing" state, an intentional exception, not the default |

**Verdict for this section: borders do essentially all of the elevation
work.** The one shadow token in active use (`shadow-xs`) is Tailwind's
smallest built-in value and is applied to exactly one primitive (`Card`).
This is squarely the "1px borders + flat surfaces" pattern common to
shadcn-based data tools, and it's the part of "sharp/professional" that
LangFuse genuinely delivers on.

## 6. Component patterns relevant to Cascaid

- **Data tables**: see §4 — compact padding, `text-xs` body / `text-sm`
  header, square corners, hairline row borders, no zebra striping.
- **Badges/status pills**: `rounded-md` (6px), never pill-shaped; a fixed
  `success`/`error`/`warning` variant set mapped to the light/dark hue pairs
  in §2, used for both discrete status badges and per-level table chips
  (`level-colors.tsx`).
- **Graph/tree visualization** (`web/src/features/trace-graph-view/`,
  file `GraphNode.tsx`): renders nodes as real DOM `<div>`s (not SVG/canvas)
  — the source comment explicitly frames this as "the win over the old
  canvas renderer" for accessibility (keyboard-focusable, `role="button"`,
  `aria-label`). Nodes get `rounded-md` corners, a 2px per-type-colored
  border with a neutral `bg-background` fill (color lives in the border/icon,
  not a colored fill), and `tabular-nums` for the observation counter
  suffix. Distinct visual language for graph "endpoint" nodes (start/end)
  vs. typed nodes: solid green/red fill with white text, vs. bordered/
  neutral-fill for everything else.
- **Sidebar/nav**: built on shadcn's standard `Sidebar` primitive
  (`ui/sidebar.tsx`, collapsible icon rail) via `AppSidebar.tsx` — white
  background in light mode (`--sidebar-background: 0 0% 100%`), differentiated
  from the content canvas only by a `sidebar-border` hairline, not a color
  block. In dark mode the sidebar becomes the single darkest surface tier.

## 7. Dark mode

Fully supported, toggled via a `.dark` class + `@custom-variant dark`
(`globals.css` L10). Not a naive light-palette inversion: see §2's dark-mode
table and the Carbon-style layering comment at `globals.css` L438–450.
`--radius` is identical (`0.5rem`) in both modes — radius is not a
theme-dependent token here.

## Mapping to Cascaid's tokens (`frontend/src/index.css`)

| Cascaid token | Current value | LangFuse equivalent | LangFuse value |
|---|---|---|---|
| `--radius-lg` | `12px` | `--radius` (`Card`, `rounded-lg`) | `8px` |
| `--radius-md` | `8px` | `--radius-md` (`Button`, `Input`, `Badge`, `rounded-md`) | `6px` |
| `--radius-sm` | `6px` | `--radius-sm` (`Button size="xs"`, `rounded-sm`) | `4px` |
| `--radius-pill` | `999px` | *(no equivalent — LangFuse badges are `rounded-md`, not pills)* | — |
| `--surface` | `#ffffff` | `--card` | `#ffffff` (identical to `--background`) |
| `--page` | `#ffffff` | `--background` | `#ffffff` |
| `--surface-muted` | `#f7f8fa` | `--muted` | `#f1f5f9` |
| `--text-primary` | `#0f1115` | `--foreground` | `#020817` |
| `--text-secondary` | `#565a6e` | `--muted-foreground` | `#64748b` |
| `--muted` | `#8b8fa3` | `--foreground-tertiary` | `#929caa` |
| `--border` | `#e4e6ec` | `--border` | `#e2e8f0` |
| `--border-strong` | `#d3d6e0` | `--border-contrast` | (`214.3 20% 80%`) |
| `--accent` | `#4f46e5` | `--primary-accent` | `#4e46e5` (near-exact match already) |
| `--shadow-sm` | `0 1px 2px rgba(15,17,21,0.04)` | Tailwind `shadow-xs` (Card only) | `0 1px 2px 0 rgb(0 0 0 / 0.05)` |
| `--shadow-md` | two-layer, ~0.04–0.06 alpha | *(no LangFuse equivalent — they don't use a second shadow tier)* | — |
| `--status-good` / `--status-warning` / `--status-serious` / `--status-critical` | green/amber/orange/red | `--light-green`/`--dark-green`, `--light-yellow`/`--dark-yellow`, *(no orange/"serious" tier)*, `--light-red`/`--dark-red` | see §2 table |
| `--node-agent`/`--node-tool`/`--node-model`/`--node-vector` | blue/green/amber/gray | `TYPE_BORDER_CLASS` per-type border colors in `GraphNode.tsx` (purple=AGENT, orange=TOOL, amber=EMBEDDING, teal=RETRIEVER, etc.) | different specific hues, same "border carries type identity, fill stays neutral" pattern |
| PipelineGraph `rx` values (canvas `rx=14`, node `rx=10`, badge `rx=8`, dot `rx=3.5`, edge label `rx=6`) | 14/10/8/6/3.5 | `GraphNode` uses a single flat `rounded-md` (6px) for the whole node box | `6px` |

## Where this conflicts with "sharp edges exactly"

Be deliberate about this rather than assuming it away:

1. **LangFuse's real radius is 4–8px, not 0.** If Cascaid genuinely wants
   zero/near-zero corners, that is a *departure* from LangFuse, not a match
   to it. "Match LangFuse exactly" and "sharp edges" are two different asks
   wherever radius is concerned — worth a decision the user makes on
   purpose. A reasonable middle path that still reads as "sharper than
   LangFuse" while nodding to it: compress Cascaid's existing 6/8/12px scale
   down toward LangFuse's 4/6/8px scale rather than to 0, and drop
   `--radius-pill` (LangFuse has no pill token — its badges, toggles, and
   equivalent chips are all `rounded-md`).
2. **Cards aren't flat — they carry a (very faint) shadow.** If Cascaid
   wants literal border-only elevation with zero shadow, that's again a
   *sharper-than-LangFuse* choice, not a faithful port. LangFuse's
   `shadow-xs` is small enough (5% opacity, 2px blur) that it's easy to miss
   visually, but it is there in the source on every `Card`.
3. **PipelineGraph's per-shape radii (14/10/8/6/3.5) have no LangFuse
   analog to size against** — LangFuse's graph nodes use one single radius
   value for the whole node box, not a per-element scale. Any translation
   to LangFuse's system means picking one of Cascaid's radii (probably the
   node/badge shape) to represent as `rounded-md`-equivalent (6px) and
   either flattening the rest to match or keeping a smaller Cascaid-specific
   scale layered on top — LangFuse's source doesn't settle this for us.
4. **No orange/"serious" severity tier exists in LangFuse's status
   language.** Its hue pairs are red/yellow/green (+ blue/violet/teal for
   entity-type score coding, not severity). Cascaid's four-tier status ramp
   (`good`/`warning`/`serious`/`critical`) has no direct LangFuse
   counterpart at the "serious" step — that tier will need an invented
   value in the same style rather than a borrowed one.
