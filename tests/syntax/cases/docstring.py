def is_docstring(node, parent):
    "Hi"
    # Must be a string
    if node.type != "string":
        return False
    # First statement in module, function, or class
    if parent.type in ("module", "function_definition", "class_definition"):
        first_stmt = parent.child_by_field_name("body").children[0]
        return first_stmt == node
    return False
