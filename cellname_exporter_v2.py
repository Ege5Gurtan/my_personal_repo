import pya

def sanitize_name(name):
    # Sanitize the cell name to ensure it's a valid XML tag name
    # For simplicity, this example just replaces invalid characters with underscores
    # and ensures the name does not start with a digit.
    sanitized = "".join(c if c.isalnum() or c in ['_', '-'] else "_" for c in name)
    if sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized

def write_cell_to_xml(cell, xml_file, level=0):
    # Sanitize cell name for XML
    cell_name_sanitized = sanitize_name(cell.name)
    
    # Indentation for hierarchy visualization
    indent = "  " * level
    xml_file.write(f"{indent}<{cell_name_sanitized}>\n")
    
    # Write instances (references to other cells)
    for inst in cell.each_inst():
        child_cell_name_sanitized = sanitize_name(inst.cell.name)
        xml_file.write(f"{indent}  <{child_cell_name_sanitized}/>\n")
    
    # Recurse into child cells for hierarchy
    for inst in cell.each_inst():
        write_cell_to_xml(inst.cell, xml_file, level + 1)
    
    xml_file.write(f"{indent}</{cell_name_sanitized}>\n")

def export_layout_to_xml(layout, output_xml_file):
    with open(output_xml_file, 'w') as xml_file:
        xml_file.write("<layout>\n")
        top_cells = layout.top_cells()
        for top_cell in top_cells:
            write_cell_to_xml(top_cell, xml_file)
        xml_file.write("</layout>\n")

# Load layout
layout = pya.Layout()
layout.read("your_layout_file.gds")

# Export layout to XML
export_layout_to_xml(layout, "layout_export.xml")
