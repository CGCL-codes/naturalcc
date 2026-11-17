"""Diversified Contract-3 mutations in existing RTOS business functions.

The immutable upstream trees and build contexts come from v6.  Every case
changes an existing operation; neither standalone fixture functions nor labels
are placed in the student workspace.  The mechanism is recorded outside it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from code_agent.scripts import build_contract3_engineering_v6 as previous
from code_agent.security_c_rules import parse_source, walk, text, declared_name, call_parts
from code_agent.security_c_macros import harvest_project_macros

DATA = ROOT / "artifacts/contract3/engineering-v7"
REVISION = "engineering-v7-20261118-r4"
SUITES, CATEGORIES = previous.SUITES, previous.CATEGORIES
PER_LABEL, SITE_COUNT = 20, 40
digest, read_json, write_json, require = (
    previous.digest, previous.read_json, previous.write_json, previous.require
)

ARRAY_FORMS = (
    "extent_index", "negative_index", "index_assignment", "unsigned_underflow",
    "loop_exit_index", "pointer_alias", "inclusive_guard", "signed_arithmetic",
    "byte_element_units", "branch_index",
)
NULL_FORMS = (
    "literal_null", "alias_overwrite", "local_null_initialization",
    "wrong_pointer_guard", "conditional_assignment", "alias_chain",
    "loop_reset", "comma_assignment", "conditional_pointer", "library_return",
)
STRING_FORMS = (
    "terminator_off_by_one", "negative_destination", "destination_alias",
    "inclusive_length_guard", "unsigned_remaining", "tail_copy",
    "source_size_copy", "loop_copy_boundary", "length_assignment", "branch_length",
)
FORMS = dict(array_oob=ARRAY_FORMS, null_pointer=NULL_FORMS,
             string_overflow=STRING_FORMS)

# Observed from the pinned Clang 18 host-analysis toolchain. RT-Thread's
# configured Cortex-M4 and Zephyr's native64 profile are both little-endian
# with binary IEEE double. Real compiler definitions take precedence.
TOOLCHAIN_MACROS = {
    '__GNUC__': '4', '__GNUC_MINOR__': '2', '__GNUC_PATCHLEVEL__': '1',
    '__ORDER_BIG_ENDIAN__': '4321', '__ORDER_LITTLE_ENDIAN__': '1234',
    '__BYTE_ORDER__': '__ORDER_LITTLE_ENDIAN__', '__FLT_RADIX__': '2',
    '__CHAR_BIT__': '8', '__SIZEOF_LONG__': '8', '__SIZEOF_LONG_LONG__': '8',
    '__SIZEOF_POINTER__': '8',
    'FLT_RADIX': '2', 'DBL_MANT_DIG': '53', 'DBL_MAX_EXP': '1024',
}


def toolchain_header():
    return ''.join(f'#ifndef {name}\n#define {name} {value}\n#endif\n'
                   for name, value in TOOLCHAIN_MACROS.items()).encode()


def enrich(site, raw, nodes):
    """Find the real containing statement; preparation retains its native use."""
    node = nodes.get((site["start"], site["end"]))
    if node is None:
        return None
    statement = node
    while statement.parent is not None and statement.parent.type != "compound_statement":
        statement = statement.parent
        if statement.type == "function_definition":
            return None
    if statement.type not in ("expression_statement", "return_statement",
                              "if_statement", "for_statement", "while_statement"):
        return None
    function = node
    while function.type != "function_definition":
        function = function.parent
    # Names are ordinary locals, chosen without shadowing an upstream symbol.
    names = {}
    occupied = set(re.findall(r"[A-Za-z_]\w*", text(function)))
    for stem in ("offset", "limit", "cursor", "other", "remaining", "position",
                 "rendered", "segment", "written"):
        candidate = stem
        while candidate in occupied:
            candidate += "_"
        occupied.add(candidate)
        names[stem] = candidate
    pointer_type = None
    # Reuse native pointer types. typeof is valid GNU C, but some external
    # analyzers lose pointer-flow facts for that spelling.
    if site["kind"] == "pointer":
        declarations = [n for n in walk(function)
                        if n.type in ("declaration", "parameter_declaration")
                        and n.start_byte < site["start"]]
        declarations += [n for n in nodes.values()
                         if n.type == "declaration" and n.parent.type == "translation_unit"]
        for declaration in declarations:
            for target in declaration.named_children:
                if target.type == "init_declarator":
                    target = target.child_by_field_name("declarator")
                if target is None or target.type not in ("identifier", "pointer_declarator"):
                    continue
                if declared_name(target) != site["destination"]:
                    continue
                qualifiers = " ".join(text(n) for n in declaration.named_children
                                      if n.type == "type_qualifier")
                kind = text(declaration.child_by_field_name("type"))
                stars = "*" * sum(n.type == "pointer_declarator" for n in walk(target))
                pointer_type = " ".join(v for v in (qualifiers, kind, stars) if v)
                break
            if pointer_type:
                break
        if pointer_type is None:
            return None
    # FreeRTOS compares the first listIS_CONTAINED_WITHIN argument without
    # dereferencing it. A one-past address alone is legal C; these sites must
    # form a pointer beyond one-past before they can be labelled defective.
    address_comparison = False
    ancestor = node.parent
    while ancestor is not None and ancestor != statement:
        if ancestor.type == 'call_expression':
            callee, args = call_parts(ancestor)
            if callee == 'listIS_CONTAINED_WITHIN' and args and args[0].start_byte <= node.start_byte < args[0].end_byte:
                address_comparison = True
        ancestor = ancestor.parent
    return dict(site, pointer_type=pointer_type, address_comparison=address_comparison, statement_start=statement.start_byte,
                statement_end=statement.end_byte, names=names,
                statement=raw[statement.start_byte:statement.end_byte].decode())


def select_sites(originals, macros):
    candidates = previous.available_sites(originals, macros)
    nodes = {}
    for file in {r["file"] for rows in candidates.values() for r in rows}:
        root = parse_source(originals[file].decode(), ".c", macros)
        nodes[file] = {(n.start_byte, n.end_byte): n for n in walk(root)}
    chosen, used = {}, set()
    for category in ("string_overflow", "array_oob", "null_pointer"):
        eligible = []
        for row in candidates[category]:
            key = row["file"], row["function"]
            if key in used:
                continue
            site = enrich(row, originals[row["file"]], nodes[row["file"]])
            if site is None:
                continue
            # Do not hoist a pointer declared by a for initializer out of scope.
            if category == "null_pointer" and site["statement"].lstrip().startswith("for"):
                continue
            eligible.append(site)
        rows, selected_functions, intervals = [], set(), {}
        for unique in (True, False):
            for site in eligible:
                key = site["file"], site["function"]
                if unique and key in selected_functions:
                    continue
                a, b = site["statement_start"], site["statement_end"]
                if any(a < y and x < b for x, y in intervals.get(site["file"], [])):
                    continue
                rows.append(site)
                selected_functions.add(key)
                intervals.setdefault(site["file"], []).append((a, b))
                if len(rows) == SITE_COUNT:
                    break
            if len(rows) == SITE_COUNT:
                break
        require(len(rows) == SITE_COUNT,
                f"Only {len(rows)} disjoint native statements for {category}")
        require(len(selected_functions) >= PER_LABEL,
                f"Fewer than 20 independent native functions for {category}")
        used.update(selected_functions)
        chosen[category] = rows
    return chosen


def array_mutation(site, form, defect):
    dest, cap = site["destination"], site["capacity"]
    n = site["names"]
    index, limit, cursor = n["offset"], n["limit"], n["cursor"]
    safe = f"({cap}) - 1"
    setup = []
    if form == "extent_index":
        expression = f"{dest}[{cap if defect else safe}]"
    elif form == "negative_index":
        expression = f"{dest}[{-1 if defect else 0}]"
    elif form == "index_assignment":
        bound = cap if defect else safe
        if site.get('call_derived_bound'):
            bound = f'({bound}) - strspn("x", "y")'
        setup = [f"long {index} = {bound};"]
        expression = f"{dest}[{index}]"
    elif form == "unsigned_underflow":
        setup = [f"unsigned long {index} = {0 if defect else 1};", f"--{index};"]
        expression = f"{dest}[{index}]"
    elif form == "loop_exit_index":
        setup = [f"unsigned long {index};",
                 f"for ({index} = 0; {index} < {cap}; ++{index}) {{ }}"]
        if not defect:
            setup.append(f"--{index};")
        expression = f"{dest}[{index}]"
    elif form == "pointer_alias":
        setup = [f"__typeof__({dest}[0]) *{cursor} = &({dest})[0];",
                 f"{cursor} += {cap if defect else safe};"]
        expression = f"(*{cursor})"
    elif form == "inclusive_guard":
        setup = [f"unsigned long {index} = 0;",
                 f"unsigned long {limit} = {cap};",
                 f"if ({limit} {'<=' if defect else '<'} {cap})",
                 f"    {index} = {limit};"]
        expression = f"{dest}[{index}]"
    elif form == "signed_arithmetic":
        setup = [f"long {index} = {-2 if defect else -1};", f"{index} += 1;"]
        expression = f"{dest}[{index}]"
    elif form == "byte_element_units":
        setup = [f"unsigned long {index} = sizeof({dest}){'' if defect else f' / sizeof({dest}[0]) - 1'};"]
        expression = f"{dest}[{index}]"
    elif form == "branch_index":
        setup = [f"unsigned long {index} = 0;",
                 f"if ({cap} > 0)", f"    {index} = {cap if defect else safe};"]
        expression = f"{dest}[{index}]"
    else:
        raise ValueError(form)
    if defect and site.get('address_comparison'):
        if form in ('extent_index', 'index_assignment', 'loop_exit_index', 'inclusive_guard', 'branch_index'):
            # Keep the same value-flow mechanism, but cross the legal one-past
            # pointer boundary when the native operation only compares pointers.
            expression = expression[:-1] + ' + 1]'
        elif form == 'pointer_alias':
            setup.append(f'++{cursor};')
    return setup, expression


def null_mutation(site, form, defect):
    dest, n = site["destination"], site["names"]
    cursor, other, position = n["cursor"], n["other"], n["position"]
    # typeof(pointer) would copy a const qualifier on the pointer itself.
    # The alias must remain assignable while preserving pointee qualifiers.
    declaration = f"{site['pointer_type']} {cursor}"
    setup = []
    if form == "literal_null":
        expression = f"((__typeof__({dest}))0)" if defect else dest
    elif form == "alias_overwrite":
        setup = [f"{declaration} = {dest};", f"{cursor} = {'0' if defect else dest};"]
        expression = cursor
    elif form == "local_null_initialization":
        setup = [f"{declaration} = {'0' if defect else dest};"]
        expression = cursor
    elif form == "wrong_pointer_guard":
        setup = [f"{declaration} = 0;",
                 f"if ({dest} {'==' if defect else '!='} 0)", f"    {cursor} = {dest};"]
        expression = cursor
    elif form == "conditional_assignment":
        setup = [f"{declaration} = {dest};",
                 f"if ({dest} != 0)", f"    {cursor} = {'0' if defect else dest};"]
        expression = cursor
    elif form == "alias_chain":
        setup = [f"{declaration} = {dest};", f"{site['pointer_type']} {other} = {cursor};",
                 f"{cursor} = {'0' if defect else dest};", f"{other} = {cursor};"]
        expression = other
    elif form == "loop_reset":
        setup = [f"{declaration} = {dest};",
                 f"for (unsigned int {position} = 0; {position} < 1; ++{position})",
                 f"    {cursor} = {'0' if defect else dest};"]
        expression = cursor
    elif form == "comma_assignment":
        setup = [f"{declaration} = {dest};"]
        expression = f"(({cursor} = {'0' if defect else dest}), {cursor})"
    elif form == "conditional_pointer":
        expression = f"({dest} ? (__typeof__({dest}))0 : {dest})" if defect else f"({dest} ? {dest} : (__typeof__({dest}))0)"
    elif form == "library_return":
        setup = [f"unsigned long {position} = strspn(\"x\", \"y\");"]
        expression = f"((__typeof__({dest})){position})" if defect else dest
    else:
        raise ValueError(form)
    return setup, expression


def string_mutation(site, form, defect):
    """Mutate the destination/length flow of the original write, not a toy call."""
    dest, n = site["destination"], site["names"]
    cursor, limit, offset, remaining, position = (n[k] for k in
        ("cursor", "limit", "offset", "remaining", "position"))
    is_copy = site["kind"] == "copy"
    args = list(site.get("arguments", []))
    length_index = (1 if "printf" in site.get("callee", "") or
                    site.get("callee", "").endswith("printk") else 2)
    cap = args[length_index] if is_copy else site["capacity"]
    if is_copy:
        return string_transfer(site, form, defect, length_index)
    setup = [f"char *{cursor} = (char *)({dest});",
             f"unsigned long {limit} = ({cap});"]
    expression = site["old"]
    if form == "terminator_off_by_one":
        setup.append(f"{cursor}[{limit}{'' if defect else ' - 1'}] = '\\0';")
    elif form == "negative_destination":
        setup.append(f"{cursor}[{-1 if defect else 0}] = '\\0';")
    elif form == "destination_alias":
        setup += [f"char *{n['other']} = {cursor};",
                  f"{n['other']} += {limit if defect else '0'};"]
        if is_copy:
            args[0] = n["other"]
            expression = f"{site['callee']}({', '.join(args)})"
        else:
            expression = f"{n['other']}[0]"
    elif form == "inclusive_length_guard":
        setup += [f"unsigned long {offset} = {limit};",
                  f"if ({offset} {'<=' if defect else '<'} {limit})",
                  f"    {cursor}[{offset}] = '\\0';"]
    elif form == "unsigned_remaining":
        setup += [f"unsigned long {remaining} = {0 if defect else 1};",
                  f"--{remaining};", f"{cursor}[{remaining}] = '\\0';"]
    elif form == "tail_copy":
        # strncpy writes exactly two bytes (including padding), even for a
        # short source; the safe control starts at zero with capacity >= 2.
        setup.append(f"strncpy({cursor}{f' + {limit} - 1' if defect else ''}, \"x\", 2);")
    elif form == "source_size_copy":
        amount = f'sizeof({n["other"]})'
        if site.get('call_derived_bound'):
            amount += ' + strspn("", "x")'
        setup += [f"char {n['other']}[2] = \"x\";",
                  f"strncpy({cursor}{f' + {limit} - 1' if defect else ''}, {n['other']}, {amount});"]
    elif form == "loop_copy_boundary":
        setup += [f"for (unsigned long {position} = 0; {position} {'<=' if defect else '<'} {limit}; ++{position})",
                  f"    {cursor}[{position}] = '\\0';"]
    elif form == "length_assignment":
        setup += [f"unsigned long {offset} = {limit};"]
        if not defect:
            setup.append(f"--{offset};")
        setup.append(f"{cursor}[{offset}] = '\\0';")
    elif form == "branch_length":
        setup += [f"unsigned long {offset} = 0;", f"if ({limit} > 0)",
                  f"    {offset} = {limit}{'' if defect else ' - 1'};",
                  f"{cursor}[{offset}] = '\\0';"]
    else:
        raise ValueError(form)
    return setup, expression


def string_transfer(site, form, defect, length_index):
    """Stage the native formatter/copy output and forward it to its real caller.

    The input, format arguments, destination and return value stay connected to
    the business operation. The added bounded staging transfer is the exercise;
    it is not an unused helper or a new standalone test function.
    """
    n, args = site["names"], list(site["arguments"])
    dest, bound = args[0], args[length_index]
    rendered, segment, written = n["rendered"], n["segment"], n["written"]
    limit, cursor, position = n["limit"], n["cursor"], n["position"]
    formatted = length_index == 1
    args[0], args[length_index] = rendered, f"sizeof({rendered}) - 1"
    call = f"{site['callee']}({', '.join(args)})"
    setup = [f"char {rendered}[64] = {{0}};", f"char {segment}[8] = {{0}};"]
    setup.append(f"int {written} = {call};" if formatted else call + ";")
    # The zero-initialized final byte is outside the bounded native call's
    # size-1 write. It already guarantees termination without a redundant store.
    if form == "terminator_off_by_one":
        setup += [f"strncpy({segment}, {rendered}, sizeof({segment}) - 1);",
                  f"{segment}[sizeof({segment}){'' if defect else ' - 1'}] = '\\0';"]
    elif form == "negative_destination":
        setup.append(f"strncpy({segment}{' - 1' if defect else ''}, {rendered}, 1);")
    elif form == "destination_alias":
        setup += [f"char *{cursor} = {segment};",
                  f"{cursor} += {'sizeof('+segment+')' if defect else '0'};",
                  f"strncpy({cursor}, {rendered}, 1);"]
    elif form == "inclusive_length_guard":
        setup += [f"unsigned long {limit} = sizeof({segment});",
                  f"if ({limit} {'<=' if defect else '<'} sizeof({segment}))",
                  f"    {segment}[{limit}] = '\\0';"]
    elif form == "unsigned_remaining":
        setup += [f"unsigned long {limit} = {0 if defect else 1};",
                  f"--{limit};", f"strncpy({segment}, {rendered}, {limit});"]
    elif form == "tail_copy":
        setup.append(f"strncpy({segment} + 6, {rendered}, {3 if defect else 2});")
    elif form == "source_size_copy":
        amount = f"sizeof({rendered})" if defect else f"sizeof({segment}) - 1"
        if site.get('call_derived_bound'):
            amount = f'({amount}) + strspn("", "x")'
        setup.append(f"strncpy({segment}, {rendered}, {amount});")
    elif form == "loop_copy_boundary":
        setup += [f"for (unsigned long {position} = 0; {position} {'<=' if defect else '<'} sizeof({segment}); ++{position})",
                  f"    {segment}[{position}] = {rendered}[{position}];"]
    elif form == "length_assignment":
        setup += [f"unsigned long {limit} = sizeof({segment});",
                  f"{limit} {'+=' if defect else '-='} 1;",
                  f"strncpy({segment}, {rendered}, {limit});"]
    elif form == "branch_length":
        setup += [f"unsigned long {limit} = sizeof({segment}){'' if defect else ' - 1'};",
                  f"if ({rendered}[0])", f"    ++{limit};",
                  f"strncpy({segment}, {rendered}, {limit});"]
    else:
        raise ValueError(form)
    setup += [f"unsigned long {n['remaining']} = ({bound});",
              f"if ({n['remaining']} > sizeof({segment}))",
              f"    {n['remaining']} = sizeof({segment});",
              f"memcpy({dest}, {segment}, {n['remaining']});"]
    return setup, written if formatted else dest


def patch(site, category, form, defect):
    setup, expression = dict(array_oob=array_mutation,
                             null_pointer=null_mutation,
                             string_overflow=string_mutation)[category](site, form, defect)
    statement = site["statement"]
    start = site["start"] - site["statement_start"]
    end = site["end"] - site["statement_start"]
    raw = statement.encode()
    replaced = (raw[:start] + expression.encode() + raw[end:]).decode()
    # Different operations can share a native function. Each local block owns
    # its fresh temporaries, so a later control does not redeclare their names.
    prefix = "\n".join(setup)
    body = prefix + ("\n" if prefix else "") + replaced
    new = "{\n" + "\n".join("    " + line for line in body.split("\n")) + "\n}"
    prefix_lines = prefix.count("\n") + 1 if prefix else 0
    # Stop the scoring range at the actual replaced operation, not the end of
    # a surrounding if/loop and its unrelated original business statements.
    operation_start = 1 + prefix_lines + raw[:start].count(b"\n")
    operation_end = operation_start + expression.count("\n")
    if category == 'null_pointer' and not defect:
        # Make the documented native nonnull precondition explicit in controls.
        # This checks the actual pointer before the annotated operation, rather
        # than asking an external checker to assume an unstated caller fact.
        new = f"if ({site['destination']} != 0) " + new
    elif category == 'string_overflow' and site['kind'] != 'copy' and not defect:
        # The direct-write controls require an actual buffer of at least two
        # bytes, including for the short tail/source-size copy mechanisms.
        new = f"if (({site['capacity']}) >= 2) " + new
    return new, operation_start, operation_end


def build_one(spec, checkout, distribution):
    originals, components = previous.load_originals(spec, checkout, distribution)
    macros = harvest_project_macros(previous.DATA / "balanced" / spec["id"] / "project")
    sites = select_sites(originals, macros)
    write_json(DATA / f"{spec['id']}-sites.json", sites)
    for suite_index, suite in enumerate(SUITES):
        records, edits = [], {}
        for category, rows in sites.items():
            groups = {}
            for ordinal, site in enumerate(rows):
                groups.setdefault((site["file"], site["function"]), []).append(ordinal)
            # At most one fault in any function. Extra control operations may
            # share the function; they can never mask another labelled fault.
            functions = list(groups)
            selected = [functions[(i + suite_index * 5) % len(functions)]
                        for i in range(PER_LABEL)]
            defective = {groups[key][suite_index % len(groups[key])] for key in selected}
            label_ordinals = Counter()
            for ordinal, site in enumerate(rows):
                defect = ordinal in defective
                form = FORMS[category][label_ordinals[defect] % len(FORMS[category])]
                label_ordinals[defect] += 1
                # Hard suites keep one call-derived bound per label/category:
                # the second source-size copy, and RT-Thread's second index
                # assignment. Both strspn("", "x") and strspn("x", "y") are
                # exactly zero, so labels and
                # safety do not depend on a guessed library return. This is an
                # explicit teaching difficulty variant, never a finding filter.
                hard_bound = ('hard' in suite and label_ordinals[defect] > 10 and (
                    (category == 'string_overflow' and form == 'source_size_copy') or
                    (category == 'array_oob' and form == 'index_assignment' and spec['id'] == 'rt_thread')))
                site = dict(site, call_derived_bound=hard_bound)
                new, operation_start, operation_end = patch(site, category, form, defect)
                record = dict(site, category=category, mechanism=form,
                              expected="defect" if defect else "control", new=new,
                              operation_start=operation_start, operation_end=operation_end,
                              case_id=f"{spec['id']}-{category}-{ordinal+1:02d}",
                              site_id=f"{category}-{ordinal+1:02d}")
                records.append(record)
                edits.setdefault(site["file"], []).append(record)
        contents = dict(originals)
        for file, rows in edits.items():
            raw, previous_start = contents[file], len(contents[file]) + 1
            for row in sorted(rows, key=lambda r: r["statement_start"], reverse=True):
                a, b = row["statement_start"], row["statement_end"]
                require(b <= previous_start, f"Overlapping native statements: {file}")
                raw = raw[:a] + row["new"].encode() + raw[b:]
                previous_start = a
            # Declarations are standard hosted-analysis headers, not stub APIs.
            contents[file] = b"#include <string.h>\n" + raw
        old_manifest = read_json(previous.DATA / suite / spec["id"] / "manifest.json")
        for item in old_manifest["project_files"]:
            if item["origin"] == "analysis-overlay":
                contents[item["file"]] = (previous.DATA / suite / spec["id"] /
                                         "project" / item["file"]).read_bytes()
        if spec["id"] == "freertos_kernel":
            database = json.loads(contents["compile_commands.json"])
            covered = {entry["file"] for entry in database}
            added = []
            for file in sorted(f for f in contents if f.endswith(".c") and f not in covered):
                # The original database covered seven kernel units only.
                # These are explicit host-analysis entries, not a claim that
                # the optional FTP/HTTP demos belong to the POSIX kernel build.
                database.append(dict(directory=".", file=file, arguments=[
                    "cc", "-I.", "-Iinclude", "-Iportable/ThirdParty/GCC/Posix",
                    "-IFreeRTOS-Plus/Source/FreeRTOS-Plus-CLI",
                    "-IFreeRTOS-Plus/Demo/Common/Demo_IP_Protocols/include",
                    "-DipconfigUSE_FTP=1", "-DipconfigUSE_HTTP=1", "-DffconfigTIME_SUPPORT=1",
                    "-D_POSIX_C_SOURCE=200809L", "-c", file]))
                added.append(file)
            contents["compile_commands.json"] = (json.dumps(database, indent=2) + "\n").encode()
            contents["build_context/v7-analysis-provenance.json"] = (
                json.dumps(dict(source="derived-source-audit-profile", files=added,
                    scope="Kernel/POSIX and real CLI/protocol headers; enable FTP/HTTP module paths for this source audit. Optional TCP/filesystem dependencies remain reported, not stubbed."),
                    indent=2) + "\n").encode()
        # Cppcheck does not supply GCC's predefined compiler macros. The RTOS
        # toolchain headers otherwise stop at #error before analysing functions.
        # Clang 18 exposes GCC compatibility 4.2.1; this guarded header selects
        # that real header branch for analyzers lacking compiler predefines.
        # Actual GCC/Clang already define these macros and remain unchanged.
        compiler_header = "build_context/cppcheck-toolchain.h"
        contents[compiler_header] = toolchain_header()
        database = json.loads(contents["compile_commands.json"])
        if spec['id'] == 'zephyr':
            # Real ext2 Kconfig defaults from the pinned upstream revision.
            # This per-module audit profile enables its actual array/type
            # declarations; it supplies no stub APIs or replacement structs.
            defaults = dict(CONFIG_MAX_FILES=10, CONFIG_EXT2_MAX_BLOCK_SIZE=4096,
                            CONFIG_EXT2_MAX_BLOCK_COUNT=10)
            context_files = []
            for entry in database:
                if entry['file'].startswith('subsys/fs/ext2/'):
                    entry['arguments'][1:1] = [f'-D{name}={value}' for name, value in defaults.items()]
                    context_files.append(entry['file'])
            kconfig = subprocess_read_kconfig(checkout)
            contents['build_context/v7-ext2-provenance.json'] = (json.dumps(dict(
                source='derived-source-audit-profile', upstream_kconfig='subsys/fs/ext2/Kconfig',
                kconfig_sha256=digest(kconfig), defaults=defaults, files=context_files,
                scope='Pinned upstream Kconfig defaults for ext2 source analysis, not a full board build.'), indent=2)+'\n').encode()
        for entry in database:
            entry["arguments"][1:1] = ["-include", compiler_header]
        contents["compile_commands.json"] = (json.dumps(database, indent=2) + "\n").encode()
        destination = DATA / suite / spec["id"]
        project = destination / "project"
        for file, raw in contents.items():
            path = project / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        cases = []
        parsed = {file: parse_source(contents[file].decode(), ".c", macros)
                  for file in edits}
        for row in records:
            file = row["file"]
            start = row["statement_start"]
            line = originals[file][:start].count(b"\n") + 2
            line += sum(r["new"].count("\n") -
                        originals[file][r["statement_start"]:r["statement_end"]].count(b"\n")
                        for r in edits[file] if r["statement_end"] <= start)
            root = parsed[file]
            functions = [f for f in walk(root) if f.type == "function_definition"
                         and declared_name(f.child_by_field_name("declarator")) == row["function"]
                         and f.start_point.row + 1 <= line <= f.end_point.row + 1]
            require(len(functions) == 1, f"Native function lost: {row['case_id']}")
            function = functions[0]
            case = {k: row[k] for k in ("case_id", "site_id", "file", "function",
                                       "category", "expected", "mechanism")}
            case.update(line_start=line + 1, line_end=line + row["operation_end"],
                        defect_line=line + row["operation_start"],
                        native_operation=row["old"], source_variant=row["kind"],
                        call_derived_bound=row['call_derived_bound'],
                        address_comparison=row['address_comparison'],
                        function_start=function.start_point.row + 1,
                        function_end=function.end_point.row + 1,
                        source_parse_error=function.has_error,
                        trigger="Reach this native statement under its original preconditions. Arrays have positive extent, native pointers are nonnull, and copy destinations have at least the positive original length (at least two bytes for tail copies). For branch_length, the native formatted/copied input must be nonempty. Address-of one-past elements requires the original callee to access that element; pointer-comparison variants instead cross beyond one-past. Call-derived bounds add strspn(\"\", \"x\"), which is exactly zero.",
                        control_reason="Only the annotated bounds/null operation is a safe control; unrelated native behavior is outside this label.",
                        fix="Restore the original native statement recorded in the external site recipe.")
            if row['call_derived_bound'] and row['category'] == 'array_oob':
                case['trigger'] = case['trigger'].replace('add strspn("", "x")', 'subtract strspn("x", "y")')
            cases.append(case)
        targets = [dict(path=f, lines=len(raw.splitlines()), sha256=digest(raw))
                   for f, raw in contents.items() if f.endswith(".c")]
        gt = dict(schema_version=1, dataset_revision=REVISION, scan_type="frequent_defects",
                  suite=suite, project={**spec, "components": components,
                                      "repo_dir": project.relative_to(ROOT).as_posix()},
                  source_sha256={r["path"]: r["sha256"] for r in targets},
                  detection_files=targets, cases=cases,
                  line_totals={"detection_files": sum(r["lines"] for r in targets)})
        write_json(destination / "ground_truth.json", gt)
        write_json(destination / "manifest.json", dict(
            dataset_revision=REVISION, upstream_commit=spec["commit"], components=components,
            recipe_sha256=digest((DATA / f"{spec['id']}-sites.json").read_bytes()),
            ground_truth_sha256=digest((destination / "ground_truth.json").read_bytes()),
            project_files=[dict(file=f, sha256=digest(raw),
                upstream_sha256=digest(originals[f]) if f in originals else None,
                origin="upstream" if f in originals else "analysis-overlay")
                for f, raw in contents.items()]))
        verify(destination)
        print(suite, spec["id"], len(targets), gt["line_totals"]["detection_files"],
              "lines; 60 defects + 60 controls;",
              sum(c["source_parse_error"] for c in cases), "unparsed functions", flush=True)


def subprocess_read_kconfig(checkout):
    """Read and validate the pinned module defaults used by the audit overlay."""
    import subprocess
    raw = subprocess.check_output(['git', '-C', str(checkout), 'show', 'HEAD:subsys/fs/ext2/Kconfig'])
    source = raw.decode()
    for name, value in [('MAX_FILES', 10), ('EXT2_MAX_BLOCK_SIZE', 4096), ('EXT2_MAX_BLOCK_COUNT', 10)]:
        block = source.split('config ' + name + '\n', 1)[1].split('\nconfig ', 1)[0]
        require(re.search(r'\bdefault\s+' + str(value) + r'\b', block), f'Unexpected Kconfig default: {name}')
    return raw


def verify(destination):
    gt = read_json(destination / "ground_truth.json")
    manifest = read_json(destination / "manifest.json")
    require(gt["dataset_revision"] == manifest["dataset_revision"] == REVISION, "Wrong revision")
    require(digest((destination / "ground_truth.json").read_bytes()) ==
            manifest["ground_truth_sha256"], "GT hash mismatch")
    require(digest((DATA / f"{gt['project']['id']}-sites.json").read_bytes()) ==
            manifest["recipe_sha256"], "Recipe hash mismatch")
    actual = {p.relative_to(destination / "project").as_posix(): digest(p.read_bytes())
              for p in (destination / "project").rglob("*") if p.is_file()}
    require(actual == {r["file"]: r["sha256"] for r in manifest["project_files"]},
            "Snapshot hash mismatch")
    require(gt["source_sha256"] == {r["path"]: r["sha256"] for r in gt["detection_files"]},
            "Target/hash mismatch")
    require(gt["line_totals"]["detection_files"] >= 10000, "Insufficient C scope")
    require(len(gt["cases"]) == 120, "Expected 120 operations")
    require(not any(c["source_parse_error"] for c in gt["cases"]), "Unparsed annotated function")
    counts = Counter((r["category"], r["expected"], r["mechanism"]) for r in gt["cases"])
    for category in CATEGORIES:
        for label in ("defect", "control"):
            require(sum(v for (c, e, _), v in counts.items() if c == category and e == label) == 20,
                    f"Contract quota: {category}/{label}")
            require({m for c, e, m in counts if c == category and e == label} == set(FORMS[category]),
                    f"Missing mechanism: {category}/{label}")
            require(all(counts[category, label, form] == 2 for form in FORMS[category]),
                    f"Expected two examples per mechanism: {category}/{label}")
    defects = [r for r in gt["cases"] if r["expected"] == "defect"]
    require(len({(r["file"], r["function"]) for r in defects}) == 60,
            "Multiple faults in one native function")
    require(not (destination / "project/ground_truth.json").exists(), "GT in student workspace")
    return gt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/tmp/naturalcc-contract3-web-sources"))
    parser.add_argument("--freertos-distribution", type=Path,
                        default=Path("/tmp/naturalcc-contract3-freertos-system-20241100"))
    parser.add_argument("--project", choices=["freertos_kernel", "zephyr", "rt_thread"])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    for spec in previous.base.load_manifest()["sources"]:
        if args.project and spec["id"] != args.project:
            continue
        if args.verify:
            for suite in SUITES:
                verify(DATA / suite / spec["id"])
                print(suite, spec["id"], "verified", flush=True)
        else:
            build_one(spec, args.upstream / spec["id"], args.freertos_distribution)


if __name__ == "__main__":
    main()
