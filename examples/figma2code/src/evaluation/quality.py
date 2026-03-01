"""
Code quality metrics for Figma2Code evaluation.

Analyzes responsiveness and maintainability of generated HTML/CSS:

Responsiveness (Direct Metrics)
- relative_unit_share:        Share of relative units (%, em, rem, vw/vh, fr, etc.) among layout-related CSS units.
- breakpoint_coverage_share:  Coverage ratio of breakpoint ranges (S/M/L/XL), from @media or utility prefixes.
- flex_grid_on_containers_rate: Ratio of container elements (div/section/main/...) using flex or grid.
- absolute_or_fixed_positioning_rate: Ratio of elements using position:absolute|fixed.
- has_viewport_meta:          Whether <meta name="viewport"> exists.
- has_responsive_images:      Whether <img srcset> or <picture> exists.
- uses_container_query:       Whether @container is used.

Maintainability (Direct Metrics)
- semantic_tag_share:             Share of semantic tags (header/section/article, etc.).
- average_dom_depth:              Average DOM depth (root=0).
- normalized_dom_depth:           min(average_dom_depth / dmax, 1.0), higher means deeper.
- inline_style_rate:              Ratio of elements with inline style.
- duplicate_declaration_share:    Share of duplicate CSS declarations (total occurrences of same (prop,value) >= 2 / total declarations).
- custom_class_coverage_rate:     Ratio of elements with custom (non-utility) classes.
- custom_class_reuse_rate:        Ratio of custom classes reused in >= 2 places.
- average_selector_complexity:    Average selector complexity (combinators/attribute selectors/pseudo-classes are scored).
- normalized_selector_complexity: min(average_selector_complexity / cmax, 1.0), higher means more complex.
- arbitrary_value_usage_rate:     ratio of class tokens using arbitrary value syntax `[...]` (e.g., w-[123px], bg-[#ff0]).

Dependencies: beautifulsoup4, lxml, tinycss2
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Optional
from collections import Counter
from functools import lru_cache

from bs4 import BeautifulSoup
from bs4.element import Tag
from tinycss2 import parse_stylesheet, parse_declaration_list
from tinycss2.ast import (
    QualifiedRule, AtRule, Declaration, FunctionBlock,
    PercentageToken, DimensionToken
)

from ..utils.console_logger import logger

# ----------------------------- Configuration -----------------------------

RELEVANT_LAYOUT_PROPS = {
    "width","min-width","max-width","height","min-height","max-height",
    "margin","margin-left","margin-right","margin-top","margin-bottom",
    "padding","padding-left","padding-right","padding-top","padding-bottom",
    "left","right","top","bottom","inset","translate","transform",
    "gap","column-gap","row-gap",
    "font-size","line-height",
    "flex","flex-basis","flex-grow","flex-shrink",
    "grid","grid-template-columns","grid-template-rows","grid-auto-columns","grid-auto-rows",
    "grid-auto-flow","grid-column","grid-row","grid-column-start","grid-column-end","grid-row-start","grid-row-end",
    "inline-size","block-size","max-inline-size","max-block-size",
    "display","position","aspect-ratio"
}

REL_UNITS = {"%","em","rem","vw","vh","vmin","vmax","ch","ex","lh","rlh","fr","svw","svh","lvw","lvh","dvw","dvh"}
ABS_UNITS = {"px","pt","cm","mm","in","pc","q"}

SEMANTIC_TAGS = {
    "header","footer","main","nav","aside",
    "section","article","figure","figcaption",
    "h1","h2","h3","h4","h5","h6","p","blockquote","code","pre",
    "ul","ol","li","dl","dt","dd",
    "table","thead","tbody","tfoot","tr","th","td","caption","colgroup","col",
    "form","label","input","select","textarea","button","fieldset","legend","output","datalist","option","optgroup",
    "a","audio","video","picture","source","track","embed","object","param","iframe",
    "details","summary","dialog","menu","menuitem","template","slot",
    "time","mark","cite","abbr","address","bdi","bdo","data","dfn","kbd","q","samp","small", "strong","sub","sup","var","wbr"
}

CONTAINER_TAGS = {"div","section","main","header","footer","aside","article","nav","form","fieldset","figure","details","dialog","template"}

TW_BP_PREFIXES = ("sm:","md:","lg:","xl:","2xl:")
BS_BP_PREFIXES = ("sm-","md-","lg-","xl-","xxl-")
TW_BP_TO_BAND  = {"sm:":"M","md:":"M","lg:":"L","xl:":"XL","2xl:":"XL"}
BS_BP_TO_BAND  = {"sm-":"S","md-":"M","lg-":"L","xl-":"XL","xxl-":"XL"}

ARBITRARY_CLASS_RE = re.compile(r'\[[^\]]+\]')

# Common CSS framework utility class patterns
UTILITY_PATTERNS = [
    # Tailwind CSS patterns
    re.compile(r'^(flex|grid|inline-flex|inline-grid|block|inline-block|inline|hidden|table|flow-root)$'),
    re.compile(r'^(items|justify|content|place|self|gap|space)-'),
    re.compile(r'^[pmwh][trblxy]?-\d+(\.\d+)?$'),
    re.compile(r'^(min-w|max-w|min-h|max-h|w|h)-(full|screen|min|max|fit|\d+)'),
    re.compile(r'^(text|bg|border|ring|divide|outline|shadow|opacity|blur|brightness|contrast|grayscale|hue-rotate|invert|saturate|sepia|backdrop)-'),
    re.compile(r'^(rounded|font|leading|tracking|z|order|col|row|auto|transition|duration|delay|ease|animate)-'),
    re.compile(r'^(static|fixed|absolute|relative|sticky)$'),
    re.compile(r'^(visible|invisible|collapse)$'),
    re.compile(r'^(overflow|overscroll|scroll|touch|select|resize|snap|appearance|cursor|pointer-events|will-change)-'),
    re.compile(r'^(hover|focus|active|visited|target|first|last|odd|even|disabled|checked|indeterminate|default|required|valid|invalid|in-range|out-of-range|placeholder|autofill|read-):'),
    # Bootstrap patterns
    re.compile(r'^(container|container-fluid|row|col|d-|flex-|justify-content-|align-|text-|bg-|border-|rounded-|shadow-|position-|top-|bottom-|start-|end-|translate-|m-|p-|mt-|mb-|ms-|me-|mx-|my-|pt-|pb-|ps-|pe-|px-|py-|w-|h-|mw-|mh-|vw-|vh-|min-vw-|min-vh-|btn|btn-|alert|alert-|badge|badge-|card|card-)'),
    # Bulma patterns
    re.compile(r'^(is-|has-|column|columns|level|media|hero|section|footer|navbar|panel|message|notification|progress|table|tag|title|subtitle|content|delete|box|button|container|icon|image|notification|progress|table|tag|title|subtitle)')
]


# ----------------------------- Utility Functions -----------------------------

def read_text(p: Path) -> str:
    """Safely read a text file, handling encoding issues."""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read {p} with UTF-8, trying latin-1: {e}")
        try:
            return p.read_text(encoding="latin-1", errors="ignore")
        except Exception as e2:
            logger.error(f"Failed to read {p}: {e2}")
            return ""


def normalize_ws(s: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r'\s+', ' ', s.strip())


def clamp01(x: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, x))


def is_container(el: Tag) -> bool:
    """Detect if an element is a container."""
    if not isinstance(el, Tag):
        return False
    
    # Basic container tags
    if el.name in CONTAINER_TAGS:
        return True
    
    # Check for child elements
    has_children = any(isinstance(c, Tag) for c in el.children)
    if not has_children:
        return False
    
    # Check if inline style contains flex/grid
    style = el.get("style", "")
    if style and re.search(r'display\s*:\s*(flex|grid|inline-flex|inline-grid)', style, re.I):
        return True
    
    # Check if class contains obvious container classes
    classes = el.get("class", [])
    if any(c in ["container", "wrapper", "layout", "row", "column", "columns"] for c in classes):
        return True
    
    return False


def parse_css_rules(css_text: str):
    """
    Parse CSS rules from text.
    
    Yields tuples of (selector_text, [Declaration], media_condition, at_rule_name).
    """
    try:
        sheet = parse_stylesheet(css_text, skip_whitespace=True, skip_comments=True)
    except Exception as e:
        logger.warning(f"Failed to parse CSS: {e}")
        return
    
    def extract_declarations(content):
        if not content:
            return []
        return [x for x in content if isinstance(x, Declaration) and x.name]
    
    def walk_rules(nodes, media_cond=None, at_rule_name=None):
        for node in nodes:
            if isinstance(node, QualifiedRule):
                try:
                    selector = "".join(t.serialize() for t in (node.prelude or [])).strip()
                    if selector:
                        yield selector, extract_declarations(node.content), media_cond, at_rule_name
                except Exception as e:
                    logger.debug(f"Failed to process rule: {e}")
                    
            elif isinstance(node, AtRule):
                keyword = (node.at_keyword or "").lower()
                if keyword in ["media", "container", "supports", "layer"]:
                    try:
                        condition = "".join(t.serialize() for t in (node.prelude or [])).strip()
                        yield from walk_rules(node.content or [], condition, keyword)
                    except Exception as e:
                        logger.debug(f"Failed to process @{keyword}: {e}")
    
    yield from walk_rules(sheet)


def extract_width_queries(cond: str) -> List[Tuple[str, float, str]]:
    """Extract width queries from media query conditions."""
    if not cond:
        return []
    
    results = []
    pattern = r'(min|max)-(?:device-)?width\s*:\s*([0-9.]+)\s*([a-zA-Z%]+)'
    for kind, val, unit in re.findall(pattern, cond, re.I):
        try:
            results.append((kind.lower(), float(val), unit.lower()))
        except ValueError:
            pass
    return results


def to_px(v: float, unit: str, base_px: float = 16.0) -> Optional[float]:
    """Convert various units to pixel values."""
    unit = (unit or "").lower().strip()
    
    conversions = {
        "px": 1.0,
        "rem": base_px,
        "em": base_px,
        "in": 96.0,
        "cm": 37.7952755906,
        "mm": 3.77952755906,
        "pt": 1.3333333333,
        "pc": 16.0,
        "q": 0.9448818898,
        "ex": base_px * 0.5,
        "ch": base_px * 0.5,
    }
    
    return conversions.get(unit, None) and v * conversions[unit]


def iter_value_tokens(tokens):
    """Recursively iterate through CSS value tokens."""
    stack = list(tokens) if tokens else []
    while stack:
        token = stack.pop(0)
        if isinstance(token, FunctionBlock):
            stack = list(token.arguments) + stack
        yield token


def count_units_in_value(prop: str, raw: str) -> Tuple[int, int]:
    """Count relative and absolute units in a CSS value."""
    rel_count = abs_count = 0
    
    # Try precise parsing with tinycss2
    try:
        tokens = parse_declaration_list(f"{prop}:{raw};", skip_whitespace=True, skip_comments=True)
        for decl in tokens:
            if isinstance(decl, Declaration) and decl.value:
                for token in iter_value_tokens(decl.value):
                    if isinstance(token, PercentageToken):
                        rel_count += 1
                    elif isinstance(token, DimensionToken):
                        unit = (token.unit or "").lower()
                        if unit in REL_UNITS:
                            rel_count += 1
                        elif unit in ABS_UNITS:
                            abs_count += 1
                return rel_count, abs_count
    except Exception:
        pass
    
    # Fallback to regex matching
    cleaned = re.sub(r'(rgb|rgba|hsl|hsla|calc|var|url|attr|clamp|min|max)\([^)]*\)', '', raw, flags=re.I)
    
    for match in re.finditer(r'(-?\d*\.?\d+)\s*([a-zA-Z%]+)(?![a-zA-Z0-9-])', cleaned):
        unit = match.group(2).lower()
        if unit in REL_UNITS:
            rel_count += 1
        elif unit in ABS_UNITS:
            abs_count += 1
    
    return rel_count, abs_count


@lru_cache(maxsize=512)
def selector_complexity(sel: str) -> int:
    """Calculate CSS selector complexity (cached)."""
    s = sel.strip()
    if not s:
        return 0
    
    complexity = 0
    
    # Remove string content
    s_cleaned = re.sub(r'"[^"]*"', '""', s)
    s_cleaned = re.sub(r"'[^']*'", "''", s_cleaned)
    
    # Combinators
    complexity += len(re.findall(r'[>+~]', s_cleaned))
    
    # Descendant selectors
    s_normalized = re.sub(r'^\s+|\s+$', '', s_cleaned)
    s_normalized = re.sub(r'\s*[>+~]\s*', '>', s_normalized)
    if ' ' in s_normalized:
        complexity += s_normalized.count(' ')
    
    # Attribute selectors
    complexity += len(re.findall(r'\[[^\]]+\]', s_cleaned))
    
    # Pseudo-classes and pseudo-elements
    complexity += len(re.findall(r'::?[a-zA-Z-]+(?:\([^)]*\))?', s_cleaned))
    
    return complexity


@lru_cache(maxsize=1024)
def is_utility_class(token: str) -> bool:
    """Detect if a class is a CSS framework utility class."""
    if not token:
        return False
    
    # Quick check for common patterns
    if ':' in token:
        return True
    
    # Check against utility patterns
    for pattern in UTILITY_PATTERNS:
        if pattern.match(token):
            return True
    
    # Check for arbitrary value syntax
    if ARBITRARY_CLASS_RE.search(token):
        return True
    
    return False


def _empty_result(html_path: str) -> Dict[str, Any]:
    """Return an empty result structure."""
    return {
        "files": {"html": html_path, "css_files": []},
        "setup": {"bands_px": {"S_max": 0, "M_max": 0, "L_max": 0}, "dmax": 0, "cmax": 0, "dup_exclude_names": []},
        "counts": {
            "elements": 0, "semantic_elements": 0, "containers": 0,
            "flex_on_containers": 0, "grid_on_containers": 0, "abspos_elements": 0,
            "unit_hits_relative": 0, "unit_hits_absolute": 0, "covered_bands": [],
            "selectors_seen": 0, "avg_selector_complexity": 0.0,
            "decl_total_considered": 0, "decl_duplicates_counted": 0,
            "elements_with_inline_style": 0, "elements_with_custom_class": 0,
            "custom_class_unique": 0, "custom_class_reused": 0,
            "avg_dom_depth": 0.0, "total_class_tokens": 0, "arbitrary_class_tokens": 0
        },
        "flags": {"has_viewport_meta": False, "has_responsive_image": False, "has_container_query": False},
        "responsiveness": {
            "relative_unit_share": 0.0, "breakpoint_coverage_share": 0.0,
            "flex_grid_on_containers_rate": 0.0, "absolute_or_fixed_positioning_rate": 0.0,
            "has_viewport_meta": False, "has_responsive_images": False, "uses_container_query": False
        },
        "maintainability": {
            "semantic_tag_share": 0.0, "average_dom_depth": 0.0, "normalized_dom_depth": 0.0,
            "inline_style_rate": 0.0, "duplicate_declaration_share": 0.0,
            "custom_class_coverage_rate": 0.0, "custom_class_reuse_rate": 0.0,
            "average_selector_complexity": 0.0, "normalized_selector_complexity": 0.0,
            "arbitrary_value_usage_rate": 0.0
        }
    }


# ----------------------------- Core Analysis Function -----------------------------

def analyze(
    html_path: Path,
    assets_dir: Optional[Path] = None,
    bands: Tuple[int, int, int] = (640, 1024, 1440),
    dmax: int = 10,
    cmax: int = 6,
    dup_exclude_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze the responsiveness and maintainability of an HTML file.
    
    Args:
        html_path: Path to the HTML file
        assets_dir: Path to the assets directory (CSS files, etc.)
        bands: Responsive breakpoint pixel values (S_max, M_max, L_max)
        dmax: Maximum expected DOM depth (for normalization)
        cmax: Maximum expected selector complexity (for normalization)
        dup_exclude_names: Keywords in filenames to exclude for duplicate calculation
    
    Returns:
        Dictionary containing analysis results with keys:
        - files: Input file information
        - setup: Analysis configuration
        - counts: Raw count values
        - flags: Boolean flags
        - responsiveness: Responsiveness metrics [0-1]
        - maintainability: Maintainability metrics [0-1]
    """
    
    dup_exclude_names = [n.lower() for n in (dup_exclude_names or [])]
    
    html = read_text(html_path)
    if not html:
        return _empty_result(str(html_path))
    
    soup = BeautifulSoup(html, "lxml")
    
    all_els: List[Tag] = soup.find_all(True)
    for i, el in enumerate(all_els):
        el.__dict__["_rid"] = i
    n_els = len(all_els)
    
    if n_els == 0:
        return _empty_result(str(html_path))
    
    # ---- Basic Maintainability ----
    # Semantic tag share
    sem_count = 0
    for el in all_els:
        if el.name in SEMANTIC_TAGS:
            if el.name == "a":
                if el.has_attr("href"):
                    sem_count += 1
            else:
                sem_count += 1
    semantic_tag_share = (sem_count / n_els) if n_els else 0.0
    
    # DOM depth
    depths = []
    def dfs(node: Tag, d: int):
        if isinstance(node, Tag):
            depths.append(d)
            for ch in node.children:
                if isinstance(ch, Tag):
                    dfs(ch, d + 1)
    
    root = soup.body if soup.body else soup
    dfs(root, 0)
    average_dom_depth = (sum(depths) / len(depths)) if depths else 0.0
    normalized_dom_depth = min(1.0, average_dom_depth / float(dmax)) if dmax > 0 else 1.0
    
    # Inline style rate
    inline_style_rate = (sum(1 for el in all_els if el.has_attr("style")) / n_els) if n_els else 0.0
    
    # Custom class & arbitrary value usage rate
    elements_with_custom = 0
    custom_class_freq = Counter()
    all_class_tokens: List[str] = []
    
    for el in all_els:
        toks = el.get("class") or []
        all_class_tokens.extend(toks)
        custom = [t for t in toks if not is_utility_class(t)]
        if custom:
            elements_with_custom += 1
            for c in custom:
                custom_class_freq[c] += 1
    
    custom_class_coverage_rate = (elements_with_custom / n_els) if n_els else 0.0
    if custom_class_freq:
        reused = sum(1 for _, f in custom_class_freq.items() if f >= 2)
        custom_class_reuse_rate = reused / len(custom_class_freq)
    else:
        custom_class_reuse_rate = 0.0
    
    total_class_tokens = len(all_class_tokens)
    arbitrary_class_tokens = sum(1 for t in all_class_tokens if ARBITRARY_CLASS_RE.search(t or ""))
    arbitrary_value_usage_rate = (arbitrary_class_tokens / total_class_tokens) if total_class_tokens else 0.0
    
    # Basic responsive flags
    has_viewport_meta = bool(soup.find("meta", attrs={"name": "viewport"}))
    has_responsive_images = bool(soup.find("img", attrs={"srcset": True})) or bool(soup.find("picture"))
    
    # ---- Collect CSS ----
    css_texts: List[Tuple[str, str]] = []
    for st in soup.find_all("style"):
        if st.string:
            css_texts.append(("[inline-style]", st.string))
    
    for ln in soup.find_all("link", rel=True, href=True):
        rels = [r.lower() for r in (ln.get("rel") or [])]
        if "stylesheet" not in rels:
            continue
        href = ln["href"]
        if href.startswith(("http://", "https://", "//")):
            continue
        p = Path(href)
        if not p.is_absolute():
            base = assets_dir if assets_dir else html_path.parent
            p = (base / p).resolve()
        if p.exists():
            content = read_text(p)
            if content:
                css_texts.append((p.name, content))
    
    # ---- CSS Analysis ----
    S_MAX, M_MAX, L_MAX = bands
    bands_set = {"S", "M", "L", "XL"}
    
    def bucket(px: float):
        if px is None:
            return None
        if px <= S_MAX:
            return "S"
        if px <= M_MAX:
            return "M"
        if px <= L_MAX:
            return "L"
        return "XL"
    
    rel_hits = abs_hits = 0
    covered_bands: Set[str] = set()
    
    container_ids = {el._rid for el in all_els if is_container(el)}
    flex_on_containers: Set[int] = set()
    grid_on_containers: Set[int] = set()
    abspos_elements: Set[int] = set()
    
    selector_complexities: List[int] = []
    decl_counter = Counter()
    total_decls = 0
    
    container_query_present = False
    
    # Process element classes and inline styles
    for el in all_els:
        rid = el._rid
        classes = set(el.get("class") or [])
        
        # Detect flex/grid from class names
        if "flex" in classes or "inline-flex" in classes:
            if rid in container_ids:
                flex_on_containers.add(rid)
        if "grid" in classes or "inline-grid" in classes:
            if rid in container_ids:
                grid_on_containers.add(rid)
        
        # Detect absolute positioning from class names
        if "absolute" in classes or "fixed" in classes:
            abspos_elements.add(rid)
        
        # Detect breakpoint prefixes
        for c in classes:
            for pfx in TW_BP_PREFIXES:
                if c.startswith(pfx):
                    b = TW_BP_TO_BAND.get(pfx)
                    if b:
                        covered_bands.add(b)
            for pfx in BS_BP_PREFIXES:
                if c.startswith(pfx):
                    b = BS_BP_TO_BAND.get(pfx)
                    if b:
                        covered_bands.add(b)
        
        # Process inline styles
        style = el.get("style")
        if style:
            for decl in [x.strip() for x in style.split(";") if ":" in x]:
                try:
                    prop, val = [y.strip() for y in decl.split(":", 1)]
                    prop = prop.lower()
                    if prop in RELEVANT_LAYOUT_PROPS:
                        r, a = count_units_in_value(prop, val)
                        rel_hits += r
                        abs_hits += a
                    if prop == "position" and re.search(r'\b(absolute|fixed)\b', val, re.I):
                        abspos_elements.add(rid)
                    if prop == "display":
                        if re.search(r'\bflex\b', val, re.I) and rid in container_ids:
                            flex_on_containers.add(rid)
                        if re.search(r'\bgrid\b', val, re.I) and rid in container_ids:
                            grid_on_containers.add(rid)
                    if prop in RELEVANT_LAYOUT_PROPS:
                        key = (prop, normalize_ws(val))
                        decl_counter[key] += 1
                        total_decls += 1
                except Exception:
                    continue
    
    # CSS rule processing
    selector_cache: Dict[str, List[Tag]] = {}
    
    for src_name, css in css_texts:
        exclude_src = any(k in src_name.lower() for k in dup_exclude_names)
        
        for selector, decls, media_cond, atname in parse_css_rules(css):
            if atname == "container":
                container_query_present = True
            
            # Process media queries
            for kind, val, unit in extract_width_queries(media_cond or ""):
                px = to_px(val, unit)
                b = bucket(px) if px is not None else None
                if b:
                    covered_bands.add(b)
            
            # Calculate selector complexity
            comp = selector_complexity(selector)
            selector_complexities.append(comp)
            
            # Match elements (using cache)
            if selector not in selector_cache:
                try:
                    if len(selector) > 500:
                        selector_cache[selector] = []
                    else:
                        selector_cache[selector] = soup.select(selector)
                except Exception:
                    selector_cache[selector] = []
            
            matched = selector_cache[selector]
            if not matched:
                continue
            
            matched_ids = {el._rid for el in matched if isinstance(el, Tag) and hasattr(el, '_rid')}
            
            # Process declarations
            for d in decls:
                prop = (d.name or "").lower().strip()
                val = "".join(t.serialize() for t in d.value).strip()
                
                if prop in RELEVANT_LAYOUT_PROPS:
                    r, a = count_units_in_value(prop, val)
                    rel_hits += r
                    abs_hits += a
                
                if prop == "display":
                    if re.search(r'\bflex\b', val, re.I):
                        for rid in matched_ids:
                            if rid in container_ids:
                                flex_on_containers.add(rid)
                    if re.search(r'\bgrid\b', val, re.I):
                        for rid in matched_ids:
                            if rid in container_ids:
                                grid_on_containers.add(rid)
                
                if prop == "position" and re.search(r'\b(absolute|fixed)\b', val, re.I):
                    for rid in matched_ids:
                        abspos_elements.add(rid)
                
                if (not exclude_src) and prop in RELEVANT_LAYOUT_PROPS:
                    key = (prop, normalize_ws(val))
                    decl_counter[key] += 1
                    total_decls += 1
    
    # ---- Responsiveness (Direct Metrics) ----
    denom = rel_hits + abs_hits
    relative_unit_share = (rel_hits / denom) if denom > 0 else 0.0
    breakpoint_coverage_share = (len(covered_bands & bands_set) / len(bands_set)) if bands_set else 0.0
    flex_grid_on_containers_rate = (len(flex_on_containers | grid_on_containers) / max(1, len(container_ids))) if container_ids else 0.0
    absolute_or_fixed_positioning_rate = (len(abspos_elements) / n_els) if n_els > 0 else 0.0
    
    # ---- Maintainability (Direct Metrics) ----
    duplicate_declaration_share = (sum((cnt - 1) for _, cnt in decl_counter.items() if cnt >= 2) / total_decls) if total_decls > 0 else 0.0

    avg_selector_complexity = (sum(selector_complexities) / len(selector_complexities)) if selector_complexities else 0.0
    normalized_selector_complexity = min(1.0, (avg_selector_complexity / float(cmax))) if cmax > 0 else 1.0
    
    # ---- Package Output ----
    return {
        "files": {
            "html": str(html_path),
            "css_files": [name for name, _ in css_texts]
        },
        "setup": {
            "bands_px": {"S_max": bands[0], "M_max": bands[1], "L_max": bands[2]},
            "dmax": dmax,
            "cmax": cmax,
            "dup_exclude_names": dup_exclude_names
        },
        "counts": {
            "elements": n_els,
            "semantic_elements": sem_count,
            "containers": len(container_ids),
            "flex_on_containers": len(flex_on_containers),
            "grid_on_containers": len(grid_on_containers),
            "abspos_elements": len(abspos_elements),
            "unit_hits_relative": int(rel_hits),
            "unit_hits_absolute": int(abs_hits),
            "covered_bands": sorted(list(covered_bands)),
            "selectors_seen": len(selector_complexities),
            "avg_selector_complexity": avg_selector_complexity,
            "decl_total_considered": total_decls,
            "decl_duplicates_counted": int(sum((cnt - 1) for _, cnt in decl_counter.items() if cnt >= 2)),
            "elements_with_inline_style": int(inline_style_rate * n_els),
            "elements_with_custom_class": int(custom_class_coverage_rate * n_els),
            "custom_class_unique": len(custom_class_freq),
            "custom_class_reused": sum(1 for _, f in custom_class_freq.items() if f >= 2),
            "avg_dom_depth": average_dom_depth,
            "total_class_tokens": total_class_tokens,
            "arbitrary_class_tokens": arbitrary_class_tokens
        },
        "flags": {
            "has_viewport_meta": has_viewport_meta,
            "has_responsive_image": has_responsive_images,
            "has_container_query": container_query_present
        },
        "responsiveness": {
            "relative_unit_share": clamp01(relative_unit_share),
            "breakpoint_coverage_share": clamp01(breakpoint_coverage_share),
            "flex_grid_on_containers_rate": clamp01(flex_grid_on_containers_rate),
            "absolute_or_fixed_positioning_rate": clamp01(absolute_or_fixed_positioning_rate),
            "has_viewport_meta": bool(has_viewport_meta),
            "has_responsive_images": bool(has_responsive_images),
            "uses_container_query": bool(container_query_present)
        },
        "maintainability": {
            "semantic_tag_share": clamp01(semantic_tag_share),
            "average_dom_depth": float(average_dom_depth),
            "normalized_dom_depth": clamp01(normalized_dom_depth),
            "inline_style_rate": clamp01(inline_style_rate),
            "duplicate_declaration_share": clamp01(duplicate_declaration_share),
            "custom_class_coverage_rate": clamp01(custom_class_coverage_rate),
            "custom_class_reuse_rate": clamp01(custom_class_reuse_rate),
            "average_selector_complexity": float(avg_selector_complexity),
            "normalized_selector_complexity": clamp01(normalized_selector_complexity),
            "arbitrary_value_usage_rate": clamp01(arbitrary_value_usage_rate)
        }
    }
