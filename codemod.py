"""Module codemod.py."""

import ast
import os
import sys

def get_type_name(node):
    """Function get_type_name.

    Arguments
    ---------
    node : Any

    Returns
    -------
    Any
    """
    if node is None:
        return "Any"
    return ast.unparse(node)

def generate_docstring(node, existing_doc):
    """Function generate_docstring.

    Arguments
    ---------
    node : Any
    existing_doc : Any

    Returns
    -------
    Any
    """
    summary = existing_doc.split("\n")[0].strip() if existing_doc else f"Function {node.name}."
    if not summary:
        summary = f"Function {node.name}."
    
    doc_lines = [summary]
    
    # Arguments
    args = [a for a in node.args.args if a.arg not in ("self", "cls")]
    if args:
        doc_lines.append("")
        doc_lines.append("Arguments")
        doc_lines.append("---------")
        for arg in args:
            t = get_type_name(arg.annotation)
            doc_lines.append(f"{arg.arg} : {t}")
            
    # Returns
    doc_lines.append("")
    doc_lines.append("Returns")
    doc_lines.append("-------")
    ret_type = get_type_name(node.returns)
    doc_lines.append(ret_type)
    
    return "\n".join(doc_lines)

def process_file(filepath):
    """Function process_file.

    Arguments
    ---------
    filepath : Any

    Returns
    -------
    Any
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    lines = source.splitlines()
    # Sort inserts/replacements in reverse order to maintain indices
    modifications = []

    # Module docstring
    module_doc = ast.get_docstring(tree, clean=False)
    if module_doc is None:
        modifications.append((0, 0, f'"""Module {os.path.basename(filepath)}."""\n'))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            existing_doc = ast.get_docstring(node, clean=False)
            
            # Find body start for insertion
            body_start_line = node.body[0].lineno - 1
            
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_doc = generate_docstring(node, existing_doc)
                content = f'"""{new_doc}\n"""'
            else: # ClassDef
                summary = existing_doc.split("\n")[0].strip() if existing_doc else f"Class {node.name}."
                if not summary: summary = f"Class {node.name}."
                content = f'"""{summary}"""'

            if existing_doc is not None:
                # Replace existing docstring
                # Note: node.body[0] is the Constant containing the docstring
                doc_node = node.body[0]
                start = doc_node.lineno - 1
                end = doc_node.end_lineno
                modifications.append((start, end, content))
            else:
                # Insert new docstring after def line
                # We need to find the correct indentation
                # Use the indentation of the first statement in body
                first_stmt = node.body[0]
                full_line = lines[first_stmt.lineno-1]
                indent = full_line[:len(full_line) - len(full_line.lstrip())]
                content_indented = "\n".join(indent + l if l else l for l in content.splitlines())
                modifications.append((body_start_line, body_start_line, content_indented))

    if not modifications:
        return False

    # Sort modifications by start line descending
    modifications.sort(key=lambda x: x[0], reverse=True)
    
    new_lines = list(lines)
    for start, end, content in modifications:
        new_lines[start:end] = [content]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    return True

changed_files = []
for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            if process_file(path):
                changed_files.append(path)

print(f"Files changed: {len(changed_files)}")
for f in changed_files:
    print(f)
