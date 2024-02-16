import pya

def write_cell_to_xml(cell, xml_file, level=0):
    # Indentation for hierarchy visualization
    indent = "  " * level
    xml_file.write(f"{indent}<cell name=\"{cell.name}\">\n")
    
    # Write instances (references to other cells)
    for inst in cell.each_inst():
        child_cell = inst.cell
        xml_file.write(f"{indent}  <instance cell=\"{child_cell.name}\"/>\n")
    
    # Recurse into child cells for hierarchy
    for inst in cell.each_inst():
        write_cell_to_xml(inst.cell, xml_file, level + 1)
    
    xml_file.write(f"{indent}</cell>\n")

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
