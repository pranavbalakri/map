import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DJI_METADATA_FIELDS = [
    'GpsLatitude', 'GpsLongitude', 'AbsoluteAltitude', 
    'RelativeAltitude', 'GimbalRollDegree', 'GimbalYawDegree',
    'GimbalPitchDegree', 'FlightRollDegree', 'FlightYawDegree',
    'FlightPitchDegree'
]

XML_NAMESPACES = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'drone-dji': 'http://www.dji.com/drone-dji/1.0/'
}

def extract_xmp_data(image_path: str) -> Optional[bytes]:
    """
    Extract XMP data from image file
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        Optional[bytes]: XMP data if found, None otherwise
    """
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
            xmp_str = b""
            
            while data:
                xmp_start = data.find(b'<x:xmpmeta')
                if xmp_start == -1:
                    break
                xmp_end = data.find(b'</x:xmpmeta')
                if xmp_end == -1:
                    break
                xmp_str += data[xmp_start:xmp_end + 12]
                data = data[xmp_end + 12:]
            
            return xmp_str if xmp_str else None
    except IOError as e:
        logger.error(f"Failed to read file {image_path}: {e}")
        return None

def extract_dji_metadata(image_path: str) -> Dict:
    """
    Extracts DJI metadata from image EXIF data
    
    Args:
        image_path (str): Path to DJI image file
        
    Returns:
        Dict: Dictionary containing DJI metadata fields
    """
    metadata = {}
    
    xmp_data = extract_xmp_data(image_path)
    if not xmp_data:
        logger.warning(f"No XMP data found in {image_path}")
        return metadata
        
    try:
        root = ET.fromstring(xmp_data)
        desc = root.find('.//rdf:Description', XML_NAMESPACES)
        
        if desc is not None:
            for field in DJI_METADATA_FIELDS:
                attr = f'{{http://www.dji.com/drone-dji/1.0/}}{field}'
                if attr in desc.attrib:
                    try:
                        metadata[field] = float(desc.attrib[attr].strip('+'))
                    except ValueError as e:
                        logger.warning(f"Failed to convert {field} value: {e}")
        else:
            logger.warning(f"No DJI metadata found in {image_path}")
            
    except ET.ParseError as e:
        logger.error(f"Error parsing XMP data from {image_path}: {e}")
    
    return metadata

def process_folder(folder_path: str) -> List[Dict]:
    """
    Process all images in a folder and extract their DJI metadata
    
    Args:
        folder_path (str): Path to folder containing DJI images
        
    Returns:
        List[Dict]: List of metadata dictionaries for each image
    """
    if not os.path.exists(folder_path):
        logger.error(f"Folder path does not exist: {folder_path}")
        return []
        
    metadata_list = []
    image_extensions = ('.jpg', '.jpeg')
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            try:
                metadata = extract_dji_metadata(image_path)
                if metadata:
                    metadata['filename'] = filename
                    metadata_list.append(metadata)
                else:
                    logger.warning(f"No metadata extracted from {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                
    return metadata_list

def write_metadata_to_csv(metadata_list: List[Dict], output_file: str) -> bool:
    """
    Write metadata to CSV file
    
    Args:
        metadata_list (List[Dict]): List of metadata dictionaries
        output_file (str): Path to output CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not metadata_list:
        logger.warning("No metadata to write to CSV")
        return False
        
    try:
        fieldnames = ['Image', 'Latitude', 'Longitude', 'Altitude', 'Degrees_Clockwise_from_North']
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metadata in metadata_list:
                if all(key in metadata for key in ['filename', 'GpsLatitude', 'GpsLongitude', 'RelativeAltitude', 'FlightYawDegree']):
                    writer.writerow({
                        'Image': metadata['filename'],
                        'Latitude': metadata['GpsLatitude'],
                        'Longitude': metadata['GpsLongitude'],
                        'Altitude': metadata['RelativeAltitude'],
                        'Degrees_Clockwise_from_North': metadata['FlightYawDegree']
                    })
                else:
                    logger.warning(f"Skipping incomplete metadata entry for {metadata.get('filename', 'unknown file')}")
        
        logger.info(f"Metadata written to {output_file}")
        return True
        
    except IOError as e:
        logger.error(f"Failed to write CSV file {output_file}: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract DJI metadata to CSV")
    parser.add_argument("folder_path", type=str, help="Path to folder containing DJI images")
    parser.add_argument("output_file", type=str, help="Path to output CSV file")
    
    args = parser.parse_args()
    
    metadata_list = process_folder(args.folder_path)
    write_metadata_to_csv(metadata_list, args.output_file)