import os
import xml.etree.ElementTree as ET

def create_folders_from_xml(xml_file_path, output_folder):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Recursive function to create folder structure
    def create_folder_structure(elem, current_path):
        # Sanitize the folder name (remove illegal characters, etc.)
        folder_name = elem.get('name').replace("/", "_").replace("\\", "_")
        path = os.path.join(current_path, folder_name)
        
        # Create the folder if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Recurse for child elements
        for child in elem:
            create_folder_structure(child, path)

    # Start the folder creation process from the root element
    create_folder_structure(root, output_folder)

# Example usage
xml_file_path = 'layout_export.xml'  # Path to your XML file
output_folder = '/path/to/output/folder'  # Base directory for the folder structure

create_folders_from_xml(xml_file_path, output_folder)
