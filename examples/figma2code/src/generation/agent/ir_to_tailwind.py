"""
Intermediate Representation (IR) to Tailwind HTML converter.

Converts AltNode IR to Tailwind CSS HTML.

"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from html import escape as h

# --------------------------- color utils ---------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def parse_paint_color(paint: Dict[str, Any]) -> Tuple[float,float,float,float]:
    c = paint.get("color") or {}
    r = clamp01(c.get("r", 0.0))
    g = clamp01(c.get("g", 0.0))
    b = clamp01(c.get("b", 0.0))
    a = clamp01(paint.get("opacity", c.get("a", 1.0)))
    return r, g, b, a

def rgba_to_hex(rgba: Tuple[float,float,float,float]) -> str:
    r,g,b,a = rgba
    return f"#{int(round(r*255)):02X}{int(round(g*255)):02X}{int(round(b*255)):02X}"

def rgba_tailwind(prefix: str, rgba: Tuple[float,float,float,float]) -> str:
    r,g,b,a = rgba
    if a >= 0.999:
        return f"{prefix}-[{rgba_to_hex((r,g,b,a))}]"
    return f"{prefix}-[rgba({int(round(r*255))},{int(round(g*255))},{int(round(b*255))},{a:.3f})]"

# --------------------------- Tailwind Predefined Value Mappings ---------------------------

# Tailwind predefined width values (rem to px, 1rem = 16px)
TAILWIND_WIDTH_MAP = {
    0: "w-0",        # 0px
    1: "w-px",       # 1px
    4: "w-1",        # 0.25rem = 4px
    8: "w-2",        # 0.5rem = 8px
    10: "w-2.5",     # 0.625rem = 10px
    12: "w-3",       # 0.75rem = 12px
    14: "w-3.5",     # 0.875rem = 14px
    16: "w-4",       # 1rem = 16px
    20: "w-5",       # 1.25rem = 20px
    24: "w-6",       # 1.5rem = 24px
    28: "w-7",       # 1.75rem = 28px
    32: "w-8",       # 2rem = 32px
    36: "w-9",       # 2.25rem = 36px
    40: "w-10",      # 2.5rem = 40px
    44: "w-11",      # 2.75rem = 44px
    48: "w-12",      # 3rem = 48px
    56: "w-14",      # 3.5rem = 56px
    64: "w-16",      # 4rem = 64px
    80: "w-20",      # 5rem = 80px
    96: "w-24",      # 6rem = 96px
    112: "w-28",     # 7rem = 112px
    128: "w-32",     # 8rem = 128px
    144: "w-36",     # 9rem = 144px
    160: "w-40",     # 10rem = 160px
    176: "w-44",     # 11rem = 176px
    192: "w-48",     # 12rem = 192px
    208: "w-52",     # 13rem = 208px
    224: "w-56",     # 14rem = 224px
    240: "w-60",     # 15rem = 240px
    256: "w-64",     # 16rem = 256px
    288: "w-72",     # 18rem = 288px
    320: "w-80",     # 20rem = 320px
    336: "w-84",     # 21rem = 336px (new)
    384: "w-96",     # 24rem = 384px
}

# Tailwind predefined height values (same as width)
TAILWIND_HEIGHT_MAP = {
    0: "h-0",        # 0px
    1: "h-px",       # 1px
    4: "h-1",        # 0.25rem = 4px
    8: "h-2",        # 0.5rem = 8px
    10: "h-2.5",     # 0.625rem = 10px
    12: "h-3",       # 0.75rem = 12px
    14: "h-3.5",     # 0.875rem = 14px
    16: "h-4",       # 1rem = 16px
    20: "h-5",       # 1.25rem = 20px
    24: "h-6",       # 1.5rem = 24px
    28: "h-7",       # 1.75rem = 28px
    32: "h-8",       # 2rem = 32px
    36: "h-9",       # 2.25rem = 36px
    40: "h-10",      # 2.5rem = 40px
    44: "h-11",      # 2.75rem = 44px
    48: "h-12",      # 3rem = 48px
    56: "h-14",      # 3.5rem = 56px
    64: "h-16",      # 4rem = 64px
    80: "h-20",      # 5rem = 80px
    96: "h-24",      # 6rem = 96px
    112: "h-28",     # 7rem = 112px
    128: "h-32",     # 8rem = 128px
    144: "h-36",     # 9rem = 144px
    160: "h-40",     # 10rem = 160px
    176: "h-44",     # 11rem = 176px
    192: "h-48",     # 12rem = 192px
    208: "h-52",     # 13rem = 208px
    224: "h-56",     # 14rem = 224px
    240: "h-60",     # 15rem = 240px
    256: "h-64",     # 16rem = 256px
    288: "h-72",     # 18rem = 288px
    320: "h-80",     # 20rem = 320px
    336: "h-84",     # 21rem = 336px (new)
    384: "h-96",     # 24rem = 384px
}

# Tailwind predefined spacing values (padding, margin, gap, etc.)
TAILWIND_SPACING_MAP = {
    0: "0",          # 0px
    1: "px",         # 1px
    4: "1",          # 0.25rem = 4px
    8: "2",          # 0.5rem = 8px
    10: "2.5",       # 0.625rem = 10px
    12: "3",         # 0.75rem = 12px
    14: "3.5",       # 0.875rem = 14px
    16: "4",         # 1rem = 16px
    20: "5",         # 1.25rem = 20px
    24: "6",         # 1.5rem = 24px
    28: "7",         # 1.75rem = 28px
    32: "8",         # 2rem = 32px
    36: "9",         # 2.25rem = 36px
    40: "10",        # 2.5rem = 40px
    44: "11",        # 2.75rem = 44px
    48: "12",        # 3rem = 48px
    56: "14",        # 3.5rem = 56px
    64: "16",        # 4rem = 64px
    80: "20",        # 5rem = 80px
    96: "24",        # 6rem = 96px
    112: "28",       # 7rem = 112px
    128: "32",       # 8rem = 128px
    144: "36",       # 9rem = 144px
    160: "40",       # 10rem = 160px
    176: "44",       # 11rem = 176px
    192: "48",       # 12rem = 192px
    208: "52",       # 13rem = 208px
    224: "56",       # 14rem = 224px
    240: "60",       # 15rem = 240px
    256: "64",       # 16rem = 256px
    288: "72",       # 18rem = 288px
    320: "80",       # 20rem = 320px
    336: "84",       # 21rem = 336px (new)
    384: "96",       # 24rem = 384px
}

# Tailwind predefined text sizes
TAILWIND_TEXT_SIZE_MAP = {
    12: "text-xs",    # 0.75rem = 12px
    14: "text-sm",    # 0.875rem = 14px
    16: "text-base",  # 1rem = 16px
    18: "text-lg",    # 1.125rem = 18px
    20: "text-xl",    # 1.25rem = 20px
    24: "text-2xl",   # 1.5rem = 24px
    30: "text-3xl",   # 1.875rem = 30px
    36: "text-4xl",   # 2.25rem = 36px
    48: "text-5xl",   # 3rem = 48px
    60: "text-6xl",   # 3.75rem = 60px
    72: "text-7xl",   # 4.5rem = 72px
    96: "text-8xl",   # 6rem = 96px
    128: "text-9xl",  # 8rem = 128px
}

# Tailwind predefined corner radius values
TAILWIND_RADIUS_MAP = {
    0: "rounded-none",     # 0px
    2: "rounded-sm",       # 0.125rem = 2px
    4: "rounded",          # 0.25rem = 4px
    6: "rounded-md",       # 0.375rem = 6px
    8: "rounded-lg",       # 0.5rem = 8px
    12: "rounded-xl",      # 0.75rem = 12px
    16: "rounded-2xl",     # 1rem = 16px
    24: "rounded-3xl",     # 1.5rem = 24px
    9999: "rounded-full",  # 9999px
}

# Tailwind predefined border widths
TAILWIND_BORDER_WIDTH_MAP = {
    0: "border-0",    # 0px
    1: "border",      # 1px
    2: "border-2",    # 2px
    4: "border-4",    # 4px
    8: "border-8",    # 8px
}

def find_closest_predefined(value: float, mapping: Dict[int, str], tolerance: float = 1.0) -> Optional[str]:
    """
    Finds the closest predefined value, returning the predefined class if the error is within tolerance.
    """
    if value is None:
        return None
    
    closest_key = None
    min_diff = float('inf')
    
    for key in mapping.keys():
        diff = abs(key - value)
        if diff < min_diff:
            min_diff = diff
            closest_key = key
    
    if closest_key is not None and min_diff <= tolerance:
        return mapping[closest_key]
    
    return None

def spacing_class(prefix: str, value: float) -> Optional[str]:
    """
    Generates the correct Tailwind spacing/positioning class based on a value and prefix.
    - Prioritizes predefined values from TAILWIND_SPACING_MAP.
    - Correctly handles string values like "px" and "2.5".
    - Falls back to arbitrary values like "pl-[17px]" if no match is found.
    """
    if value is None:
        return None
    
    predefined = find_closest_predefined(value, TAILWIND_SPACING_MAP)
    
    if predefined:
        # If the predefined value is "px", there is no hyphen between the prefix and value (e.g., "pl-px")
        # Otherwise, connect with a hyphen (e.g., "pl-4", "gap-2.5")
        separator = "" if predefined == "px" else "-"
        return f"{prefix}{separator}{predefined}"
    else:
        # Fallback to arbitrary value
        val_int = int(round(value))
        if val_int == 0 and value > 0: # Handle cases where 0 < value < 1
             return f"{prefix}-px"
        if val_int > 0:
            return f"{prefix}-[{val_int}px]"
    return None

# --------------------------- helpers ---------------------------
def dim_exact(prefix: str, v: float, precision: int = 2):
    if v is None: return None
    f = float(v)
    if f <= 0: return None
    s = f"{f:.{precision}f}".rstrip("0").rstrip(".")
    return f"{prefix}-[{s}px]"

def dim(prefix: str, v):
    """Uniformly generates Tailwind dimension classes; prioritizes predefined classes, otherwise uses arbitrary values."""
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    
    # Special handling: output 1px for 0 < v < 1
    if f > 0.0 and f < 1.0:
        return f"{prefix}-px"
    
    # Select the corresponding map based on the prefix
    if prefix == "w":
        predefined = find_closest_predefined(f, TAILWIND_WIDTH_MAP)
        if predefined:
            return predefined
    elif prefix == "h":
        predefined = find_closest_predefined(f, TAILWIND_HEIGHT_MAP)
        if predefined:
            return predefined
    
    # If no suitable predefined value is found, use an arbitrary value
    iv = int(round(f))
    if iv <= 0:
        return None
    return f"{prefix}-[{iv}px]"

def first_visible(paints: Any, kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(paints, list): return None
    for p in paints:
        if not isinstance(p, dict): continue
        if not p.get("visible", True): continue
        if kind is None or p.get("type") == kind:
            return p
    return None

def image_ref(node: Dict[str, Any]) -> Optional[str]:
    if node.get("imageRef"): return node["imageRef"]
    p = first_visible(node.get("fills"), "IMAGE")
    if p and p.get("imageRef"): return p["imageRef"]
    return None

def bbox(n: Dict[str, Any]) -> Tuple[float,float,float,float]:
    bb = n.get("absoluteBoundingBox") or {}
    return (float(bb.get("x", n.get("x", 0))), float(bb.get("y", n.get("y", 0))),
            float(bb.get("width", n.get("width", 0))), float(bb.get("height", n.get("height", 0))))

# --------------------------- tailwind helpers ---------------------------

def text_color_classes(fills: Any) -> List[str]:
    p = first_visible(fills, "SOLID")
    if not p: return []
    return [rgba_tailwind("text", parse_paint_color(p))]

def bg_color_classes(fills: Any) -> List[str]:
    p = first_visible(fills, "SOLID")
    if not p: return []
    return [rgba_tailwind("bg", parse_paint_color(p))]

def size_classes(node: Dict[str, Any], prefer_auto_for_hug: bool = True) -> List[str]:
    out: List[str] = []
    w = node.get("width"); h = node.get("height")
    hs = (node.get("layoutSizingHorizontal") or "FIXED").upper()
    vs = (node.get("layoutSizingVertical") or "FIXED").upper()
    if hs == "FILL": out.append("w-full")
    elif hs == "HUG" and prefer_auto_for_hug: out.append("w-auto")
    elif w is not None:
        tok = dim("w", w)
        if tok: out.append(tok)
    if vs == "FILL": out.append("h-full")
    elif vs == "HUG" and prefer_auto_for_hug: out.append("h-auto")
    elif h is not None:
        tok = dim("h", h)
        if tok: out.append(tok)
    
    grow = node.get("layoutGrow", 0)
    if grow > 0:
        out.append("flex-grow")
    
    return out

def text_wrap_classes(node: dict) -> list[str]:
    auto = (node.get("textAutoResize") or "").upper()
    hs   = (node.get("layoutSizingHorizontal") or "").upper()
    chars = node.get("characters") or ""
    has_nl = "\n" in chars
    w = node.get("width")

    out: list[str] = []

    if "WIDTH" in auto:
        # Horizontal hug: no wrap
        out += ["inline-block", "whitespace-pre" if has_nl else "whitespace-nowrap"]

    elif auto in ("HEIGHT", "NONE") or (not auto and hs == "FIXED"):
        # Fixed width + normal wrap (preserve manual newlines; don't break words)
        # Use integer pixel values to avoid non-standard classes like w-[123.45px]
        if w is not None and float(w) > 0:
            out.append(f"w-[{int(round(float(w)))}px]")
        out += ["whitespace-pre-line", "break-normal"]

    else:
        # Other cases: fallback to HS
        if hs == "FILL":
            out += ["w-full", "whitespace-pre-line", "break-normal"]
        elif hs == "HUG":
            out += ["inline-block", "whitespace-pre" if has_nl else "whitespace-nowrap"]
        else:
            # Fallback to integer pixels is safer
            if w is not None and float(w) > 0:
                out.append(f"w-[{int(round(float(w)))}px]")
            out += ["whitespace-pre-line", "break-normal"]

    return out


def radius_classes(node: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    cr = node.get("cornerRadius")
    rads = node.get("rectangleCornerRadii")
    
    if isinstance(cr, (int, float)):
        # Try to use predefined radius classes
        predefined = find_closest_predefined(float(cr), TAILWIND_RADIUS_MAP)
        if predefined:
            out.append(predefined)
        else:
            out.append(f"rounded-[{int(round(float(cr)))}px]")
    elif isinstance(rads, (list, tuple)) and len(rads) == 4:
        tl, tr, br, bl = rads
        if tl == tr == br == bl and tl is not None:
            # All corners are the same, use a uniform radius class
            predefined = find_closest_predefined(float(tl), TAILWIND_RADIUS_MAP)
            if predefined:
                out.append(predefined)
            else:
                out.append(f"rounded-[{int(round(float(tl)))}px]")
        else:
            # Different corners have different radius values
            corner_map = {
                "tl": tl,
                "tr": tr,
                "br": br,
                "bl": bl
            }
            for corner, value in corner_map.items():
                if value is not None:
                    # Try to find a predefined value for each corner
                    # Note: Tailwind does not have predefined classes for single-corner radius, so use arbitrary values
                    out.append(f"rounded-{corner}-[{int(round(float(value)))}px]")
    return out

def border_from_strokes(strokes: Any, node: Dict[str, Any]) -> List[str]:
    p = first_visible(strokes, "SOLID")
    if not p: return []
    out: List[str] = []
    w = node.get("strokeWeight") or 0
    if w:
        # Try to use predefined border widths
        predefined = find_closest_predefined(float(w), TAILWIND_BORDER_WIDTH_MAP)
        if predefined:
            out.append(predefined)
        else:
            out.append(f"border-[{int(round(float(w)))}px]")
    else:
        out.append("border")
    out.append(rgba_tailwind("border", parse_paint_color(p)))
    return out

def padding_classes(node: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key, prefix in (("paddingLeft", "pl"), ("paddingRight", "pr"), 
                       ("paddingTop", "pt"), ("paddingBottom", "pb")):
        v = node.get(key)
        if v is not None:
            cls = spacing_class(prefix, float(v))
            if cls:
                out.append(cls)
    return out

def gap_classes(node: Dict[str, Any], axis: Optional[str] = None) -> List[str]:
    s = node.get("itemSpacing")
    if s is None or float(s) <= 0:
        return []
    
    prefix = "gap"
    if axis in ("x", "y"):
        prefix = f"gap-{axis}"
        
    cls = spacing_class(prefix, float(s))
    return [cls] if cls else []

def text_style_classes(style: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    
    # Text size
    fs = style.get("fontSize")
    if fs is not None:
        # Try to use predefined text sizes
        predefined = find_closest_predefined(float(fs), TAILWIND_TEXT_SIZE_MAP)
        if predefined:
            out.append(predefined)
        else:
            out.append(f"text-[{int(round(float(fs)))}px]")
    
    # Line height
    lh = style.get("lineHeightPx") or style.get("lineHeight")
    if lh is not None:
        out.append(f"leading-[{int(round(float(lh)))}px]")
    
    # Font weight
    fw = style.get("fontWeight")
    if isinstance(fw, (int, float)):
        weight = int(round(float(fw) / 100) * 100)
        weight = min(900, max(100, weight))
        tw = {
            100: "thin", 200: "extralight", 300: "light", 
            400: "normal", 500: "medium", 600: "semibold", 
            700: "bold", 800: "extrabold", 900: "black"
        }[weight]
        out.append(f"font-{tw}")
    
    # Letter spacing
    ls = style.get("letterSpacing")
    if isinstance(ls, (int, float)) and abs(ls) > 0.01:
        out.append(f"tracking-[{float(ls):.2f}px]")
    
    # Text alignment
    align = style.get("textAlignHorizontal")
    if isinstance(align, str):
        a = align.upper()
        out.append({
            "CENTER": "text-center",
            "RIGHT": "text-right",
            "END": "text-right",
            "JUSTIFIED": "text-justify",
            "JUSTIFY": "text-justify"
        }.get(a, "text-left"))
    
    # Text case
    case = style.get("textCase") or style.get("textTransform")
    if isinstance(case, str):
        c = case.upper()
        if "UPPER" in c: out.append("uppercase")
        elif "LOWER" in c: out.append("lowercase")
        elif "TITLE" in c: out.append("capitalize")
    
    return out

def is_line_like(node: Dict[str, Any]) -> bool:
    w = float(node.get("width") or 0); h = float(node.get("height") or 0)
    return (h <= 1.0 and w >= 10.0) or (w <= 1.0 and h >= 10.0)

# --------------------------- generator ---------------------------

class GeneratorV7:
    def __init__(self, absolute_top_level: bool = True, sort_top_level: bool = True):
        self.absolute_top_level = absolute_top_level
        self.sort_top_level = sort_top_level
        self.detected_fonts: List[str] = []

    def generate(self, nodes: List[Dict[str, Any]], title: str = "") -> str:
        body, head_extra = self._render_top_level(nodes)
        links, font_css = self._font_links_and_css()
        return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>{h(title) if title else "Figma to Tailwind"}</title>
<script src="https://cdn.tailwindcss.com"></script>
{links}
{font_css}
<style>img{{display:block;max-width:100%;height:auto}}</style>
{head_extra}
</head>
<body class="antialiased">
{body}
</body>
</html>"""

    def _collect_fonts(self, nodes: List[Dict[str, Any]]):
        for n in nodes:
            if (n.get("type") or "").upper() == "TEXT":
                st = n.get("style") or {}
                fam = st.get("fontFamily")
                if fam: self.detected_fonts.append(fam)
                
    def _font_links_and_css(self) -> tuple[str, str]:
        fams = sorted({f for f in self.detected_fonts if isinstance(f, str)})
        links = []
        for fam in fams:
            q = fam.replace(" ", "+")
            links.append(
                f'<link href="https://fonts.googleapis.com/css2?family={q}:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">'
            )
        css = "\n".join(
            [f".font-{fam.lower().replace(' ','-')}{{font-family:'{fam}',ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,'Helvetica Neue',Arial,'Noto Sans',sans-serif}}"
            for fam in fams]
        )
        return "\n".join(links), f"<style>{css}</style>" if css else ""

    def _render_top_level(self, nodes: List[Dict[str, Any]]) -> Tuple[str,str]:
        items = [n for n in nodes if n.get("visible", True)]
        if not items: return "", ""
        
        self._collect_fonts(items)
        
        # Calculate canvas boundaries
        xs=[]; ys=[]; xe=[]; ye=[]
        for n in items:
            x,y,w,h = bbox(n); xs.append(x); ys.append(y); xe.append(x+w); ye.append(y+h)
        min_x=min(xs); min_y=min(ys); max_x=max(xe); max_y=max(ye)
        W=int(round(max_x-min_x)); H=int(round(max_y-min_y))

        # Sort
        if self.sort_top_level:
            items.sort(key=lambda n: (float((n.get("absoluteBoundingBox") or {}).get("y", n.get("y", 0))),
                                    float((n.get("absoluteBoundingBox") or {}).get("x", n.get("x", 0)))))

        # Create transparent container - use predefined classes
        w_class = dim("w", W) or f"w-[{W}px]"
        h_class = dim("h", H) or f"h-[{H}px]"
        parts: List[str] = [f'<main class="relative {w_class} {h_class} mx-auto overflow-hidden">']
        
        for n in items:
            x,y,w,h = bbox(n)
            left = int(round(x - min_x)); top = int(round(y - min_y))
            z = n.get("zIndex")
            zcls = f" z-[{int(z)}]" if isinstance(z, (int,float)) else ""
            
            # Use the new spacing_class function to get position classes
            left_cls = spacing_class("left", left) or f"left-[{left}px]"
            top_cls = spacing_class("top", top) or f"top-[{top}px]"
            
            if self.absolute_top_level:
                parts.append(f'<div class="absolute {left_cls} {top_cls}{zcls}">')
                parts.append(self._render_node(n, left, top, W))
                parts.append("</div>")
            else:
                parts.append(self._render_node(n, left, top, W))
        
        parts.append("</main>")
        return "\n".join(parts), ""

    def _render_node(self, n: Dict[str, Any], left: int = 0, top: int = 0, canvas_w: int = 0) -> str:
        t = (n.get("type") or "").upper()
        if t == "TEXT": return self._render_text(n)
        if t in {"VECTOR","BOOLEAN_OPERATION","POLYGON","STAR","LINE","ELLIPSE"}: return self._render_graphic(n)
        if t == "RECTANGLE":
            if not n.get("children") and image_ref(n): return self._render_graphic(n)
            if is_line_like(n): return self._render_line(n)
        return self._render_container(n, left, top, canvas_w)

    def _render_line(self, n: Dict[str, Any]) -> str:
        """Render line elements"""
        img = image_ref(n)
        w = int(round(float(n.get("width") or 0)))
        hpx = int(round(float(n.get("height") or 0)))
        hpx = 1 if hpx == 0 else hpx
        w = w if w > 0 else 1
        
        if img:
            # Use predefined classes
            w_class = dim("w", w) or f"w-[{w}px]"
            h_class = dim("h", hpx) or f"h-[{hpx}px]"
            cls = " ".join([w_class, h_class] + radius_classes(n))
            return f'<img src="{h(img)}" class="{cls} object-cover" alt="{h(n.get("uniqueName",""))}"/>'
        
        # Use the node's own color
        p = first_visible(n.get("fills"), "SOLID")
        if p:
            color_cls = rgba_tailwind("bg", parse_paint_color(p))
        else:
            # Default gray line
            color_cls = "bg-gray-300"
        
        if w >= hpx:
            w_class = dim("w", w) or f"w-[{w}px]"
            cls = " ".join([w_class, "h-px", color_cls])
        else:
            h_class = dim("h", hpx) or f"h-[{hpx}px]"
            cls = " ".join(["w-px", h_class, color_cls])
        
        return f'<div class="{cls}"></div>'

    def _render_text(self, n: Dict[str, Any]) -> str:
        classes: List[str] = []
        classes += text_wrap_classes(n)
        classes += text_color_classes(n.get("fills"))
        classes += text_style_classes(n.get("style", {}))
        fam = (n.get("style") or {}).get("fontFamily")
        if fam:
            classes.append(f"font-{fam.lower().replace(' ','-')}")
        
        segs = n.get("styledTextSegments")
        if isinstance(segs, list) and segs:
            parts = []
            base = n.get("style", {}) or {}
            base_color = text_color_classes(n.get("fills"))
            
            for seg in segs:
                s = dict(base)
                s.update(seg.get("style") or {})
                fam_seg = (s.get("fontFamily") or fam)
                fam_cls = f" font-{fam_seg.lower().replace(' ','-')}" if fam_seg else ""
                seg_cls = text_style_classes(s) + (base_color or [])
                parts.append(f'<span class="{" ".join(seg_cls)}{fam_cls}">{h(seg.get("text",""))}</span>')
            content = "".join(parts)
        else:
            content = h(n.get("characters","") or "")
        
        return f'<p class="{" ".join(classes)}">{content}</p>'

    def _render_graphic(self, n: Dict[str, Any]) -> str:
        img = image_ref(n)
        cls = " ".join(size_classes(n) + radius_classes(n))
        if img:
            # Correct for zero height
            h_val = float(n.get("height") or 0)
            if int(round(h_val)) == 0:
                cls = cls.replace("h-0", "h-px")
            return f'<img src="{h(img)}" class="{cls} object-contain" alt="{h(n.get("uniqueName",""))}"/>'
        
        p = first_visible(n.get("fills"), "SOLID")
        if p:
            return f'<div class="{cls} {rgba_tailwind("bg", parse_paint_color(p))}"></div>'
        return f'<div class="{cls}"></div>'

    def _render_container(self, n: Dict[str, Any], left: int, top: int, canvas_w: int) -> str:
        if not n.get("children"):
            img = image_ref(n)
            if img:
                cls = " ".join(size_classes(n) + radius_classes(n))
                return f'<img src="{h(img)}" class="{cls} object-contain" alt="{h(n.get("uniqueName",""))}"/>'
        
        layout = (n.get("layoutMode") or "NONE").upper()
        classes: List[str] = []
        classes += size_classes(n)
        classes += radius_classes(n)
        classes += bg_color_classes(n.get("fills"))
        classes += border_from_strokes(n.get("strokes"), n)
        classes += padding_classes(n)

        inner: List[str] = []
        children = n.get("children") or []
        
        if layout in {"HORIZONTAL","VERTICAL"}:
            classes.append("flex")
            classes.append("flex-row" if layout=="HORIZONTAL" else "flex-col")
            pa = (n.get("primaryAxisAlignItems") or "MIN").upper()
            ca = (n.get("counterAxisAlignItems") or "MIN").upper()
            justify = {"MIN":"justify-start","CENTER":"justify-center","MAX":"justify-end","SPACE_BETWEEN":"justify-between"}.get(pa,"justify-start")
            items = {"MIN":"items-start","CENTER":"items-center","MAX":"items-end","BASELINE":"items-baseline","STRETCH":"items-stretch"}.get(ca,"items-start")
            
            # Heuristic: if the row is on the right edge, use justify-end
            x,y,w,h = bbox(n)
            if layout=="HORIZONTAL" and (x + w >= canvas_w - 2):
                justify = "justify-end"
            
            classes.extend([justify, items])
            classes += gap_classes(n, "x" if layout=="HORIZONTAL" else "y")
            
            for c in children:
                if c.get("visible", True):
                    inner.append(self._render_node(c))
        else:
            if children:
                classes.append("relative")
                for c in children:
                    if c.get("visible", True):
                        left_c = float(c.get("x") or 0)
                        top_c = float(c.get("y") or 0)
                        
                        # Also use the new spacing_class function for child element positions
                        left_cls = spacing_class("left", left_c) or f"left-[{int(round(left_c))}px]"
                        top_cls = spacing_class("top", top_c) or f"top-[{int(round(top_c))}px]"
                        
                        inner.append(f'<div class="absolute {left_cls} {top_cls}">')
                        inner.append(self._render_node(c))
                        inner.append("</div>")
        
        return f'<div class="{" ".join([c for c in classes if c])}">{"".join(inner)}</div>'
    
# -------------- CLI --------------
def load_nodes(data: Any):
    if isinstance(data, list): return data
    if isinstance(data, dict) and "nodes" in data: return data["nodes"]
    raise ValueError("IRNode JSON must be a list or an object with 'nodes'.")

def IRtoTailwind(IRNode_json, save_path=None):
    nodes = load_nodes(IRNode_json)
    html = GeneratorV7().generate(nodes)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html
