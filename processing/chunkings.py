import fitz  # PyMuPDF
import pandas as pd
from typing import List, Dict, Any, Tuple
import re
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text chunks from PDF with page and line information."""
        logger.info(f"Starting text extraction from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        chunks = []
        total_pages = len(doc)
        logger.info(f"PDF has {total_pages} pages")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.info(f"Processing page {page_num + 1}/{total_pages} for text extraction")
            text = page.get_text()
            logger.info(f"Page {page_num + 1}: Extracted {len(text)} characters of text")
            
            # Split text into lines for better tracking
            lines = text.split('\n')
            logger.info(f"Page {page_num + 1}: Split into {len(lines)} lines")
            current_chunk = ""
            chunk_lines = []
            line_start = 0
            page_chunks = 0
            
            for i, line in enumerate(lines):
                if len(current_chunk) + len(line) > self.chunk_size and current_chunk:
                    # Save current chunk
                    page_chunks += 1
                    chunk_info = {
                        'type': 'text',
                        'content': current_chunk.strip(),
                        'page': page_num + 1,
                        'line_start': line_start + 1,
                        'line_end': line_start + len(chunk_lines),
                        'metadata': {
                            'chunk_type': 'text',
                            'page_number': page_num + 1,
                            'line_range': f"{line_start + 1}-{line_start + len(chunk_lines)}"
                        }
                    }
                    chunks.append(chunk_info)
                    logger.info(f"Page {page_num + 1}: Created text chunk {page_chunks} (lines {line_start + 1}-{line_start + len(chunk_lines)}, {len(current_chunk.strip())} chars)")
                    
                    # Start new chunk with overlap
                    overlap_text = ' '.join(chunk_lines[-3:])  # Last 3 lines as overlap
                    current_chunk = overlap_text + " " + line
                    chunk_lines = chunk_lines[-3:] + [line]
                    line_start = i - 2
                else:
                    current_chunk += " " + line if current_chunk else line
                    chunk_lines.append(line)
            
            # Add final chunk of the page
            if current_chunk.strip():
                page_chunks += 1
                chunk_info = {
                    'type': 'text',
                    'content': current_chunk.strip(),
                    'page': page_num + 1,
                    'line_start': line_start + 1,
                    'line_end': line_start + len(chunk_lines),
                    'metadata': {
                        'chunk_type': 'text',
                        'page_number': page_num + 1,
                        'line_range': f"{line_start + 1}-{line_start + len(chunk_lines)}"
                    }
                }
                chunks.append(chunk_info)
                logger.info(f"Page {page_num + 1}: Created final text chunk {page_chunks} (lines {line_start + 1}-{line_start + len(chunk_lines)}, {len(current_chunk.strip())} chars)")
        
        doc.close()
        logger.info(f"Text extraction complete: {len(chunks)} total text chunks created")
        return chunks
        
        doc.close()
        return chunks
    
    def extract_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF with page information."""
        logger.info(f"Starting image extraction from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        images = []
        total_pages = len(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.info(f"Processing page {page_num + 1}/{total_pages} for image extraction")
            
            # Get image list for the page
            image_list = page.get_images(full=True)
            logger.info(f"Page {page_num + 1}: Found {len(image_list)} images")
            
            for img_index, img in enumerate(image_list):
                try:
                    logger.info(f"Page {page_num + 1}: Processing image {img_index + 1}/{len(image_list)}")
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_data))
                        
                        # Get image position on page (approximate)
                        img_rect = page.get_image_bbox(img)
                        
                        image_info = {
                            'type': 'image',
                            'content': img_data,
                            'page': page_num + 1,
                            'image_index': img_index,
                            'position': {
                                'x0': img_rect.x0 if img_rect else 0,
                                'y0': img_rect.y0 if img_rect else 0,
                                'x1': img_rect.x1 if img_rect else 100,
                                'y1': img_rect.y1 if img_rect else 100
                            },
                            'metadata': {
                                'chunk_type': 'image',
                                'page_number': page_num + 1,
                                'image_index': img_index,
                                'size': img_pil.size,
                                'format': img_pil.format
                            }
                        }
                        images.append(image_info)
                        logger.info(f"Page {page_num + 1}: Successfully extracted image {img_index + 1} (size: {img_pil.size}, format: {img_pil.format})")
                    else:
                        logger.warning(f"Page {page_num + 1}: Skipping image {img_index + 1} (unsupported color space: {pix.n} channels)")
                    
                    pix = None
                except Exception as e:
                    logger.error(f"Error extracting image {img_index + 1} from page {page_num + 1}: {e}")
                    continue
        
        doc.close()
        logger.info(f"Image extraction complete: {len(images)} total images extracted")
        return images
    
    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using fitz (PyMuPDF)."""
        logger.info(f"Starting table extraction from PDF: {pdf_path}")
        tables = []
        
        try:
            # Open PDF with fitz
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF has {total_pages} pages for table extraction")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                logger.info(f"Processing page {page_num + 1}/{total_pages} for table extraction")
                
                # Extract tables using fitz table finder
                table_dict = page.find_tables()
                logger.info(f"Page {page_num + 1}: Table finder returned {len(table_dict.tables) if table_dict and table_dict.tables else 0} tables")
                
                if table_dict and table_dict.tables:
                    for table_index, table in enumerate(table_dict.tables):
                        logger.info(f"Page {page_num + 1}: Processing table {table_index + 1}/{len(table_dict.tables)}")
                        # Extract table content
                        table_data = []
                        table_str = ""
                        
                        # Get table rows
                        rows = table.extract()
                        logger.info(f"Page {page_num + 1}: Table {table_index + 1} has {len(rows)} rows")
                        
                        if rows and len(rows) > 1:  # At least 2 rows to be a table
                            logger.info(f"Page {page_num + 1}: Table {table_index + 1} has {len(rows)} rows, processing...")
                            # Process each row
                            for row_idx, row in enumerate(rows):
                                if any(cell and str(cell).strip() for cell in row):
                                    # Clean and format row data
                                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                                    table_data.append(clean_row)
                                    logger.debug(f"Page {page_num + 1}: Table {table_index + 1}, Row {row_idx + 1}: {len(clean_row)} columns")
                            
                            if table_data:
                                # Format table as string
                                table_str = self._format_table_as_string(table_data)
                                
                                table_info = {
                                    'type': 'table',
                                    'content': table_str,
                                    'page': page_num + 1,
                                    'table_index': table_index,
                                    'table_data': table_data,
                                    'metadata': {
                                        'chunk_type': 'table',
                                        'page_number': page_num + 1,
                                        'table_index': table_index,
                                        'rows': len(table_data),
                                        'columns': len(table_data[0]) if table_data else 0,
                                        'column_names': [f"col_{i}" for i in range(len(table_data[0]))] if table_data else [],
                                        'extraction_method': 'fitz_table_finder'
                                    }
                                }
                                tables.append(table_info)
                                logger.info(f"Page {page_num + 1}: Successfully extracted table {table_index + 1} ({len(table_data)} rows, {len(table_data[0]) if table_data else 0} columns)")
                            else:
                                logger.warning(f"Page {page_num + 1}: Table {table_index + 1} has no valid data after processing")
                        else:
                            logger.warning(f"Page {page_num + 1}: Table {table_index + 1} has insufficient rows ({len(rows) if rows else 0})")
            
            doc.close()
            
            if tables:
                logger.info(f"Successfully extracted {len(tables)} tables using fitz table finder")
            else:
                logger.warning("No tables found with fitz table finder, trying pattern matching fallback")
                tables.extend(self._extract_table_patterns(pdf_path))
                
        except Exception as e:
            logger.error(f"Error extracting tables with fitz: {e}")
            # Fallback to pattern matching
            tables.extend(self._extract_table_patterns(pdf_path))
        
        logger.info(f"Table extraction complete: {len(tables)} total tables extracted")
        return tables
    
    def _format_table_as_string(self, table_data) -> str:
        """Format table data as a readable string."""
        if not table_data:
            return ""
        
        # Find the maximum width for each column
        col_widths = []
        for col_idx in range(len(table_data[0])):
            max_width = 0
            for row in table_data:
                if col_idx < len(row):
                    cell_width = len(str(row[col_idx]))
                    max_width = max(max_width, cell_width)
            col_widths.append(max_width)
        
        # Format the table
        formatted_lines = []
        for row in table_data:
            formatted_row = []
            for col_idx, cell in enumerate(row):
                if col_idx < len(col_widths):
                    cell_str = str(cell) if cell else ""
                    formatted_cell = cell_str.ljust(col_widths[col_idx])
                    formatted_row.append(formatted_cell)
            formatted_lines.append(" | ".join(formatted_row))
        
        return "\n".join(formatted_lines)
    
    def _extract_table_patterns(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Fallback method to extract table-like patterns from text."""
        logger.info("Starting pattern matching table extraction fallback")
        try:
            import fitz
            doc = fitz.open(pdf_path)
            tables = []
            total_pages = len(doc)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                logger.info(f"Pattern matching: Processing page {page_num + 1}/{total_pages}")
                text = page.get_text()
                
                # Look for table-like patterns (rows with multiple columns separated by spaces/tabs)
                lines = text.split('\n')
                logger.info(f"Pattern matching: Page {page_num + 1} has {len(lines)} lines to analyze")
                table_lines = []
                current_table = []
                page_tables = 0
                
                for line in lines:
                    # Check if line looks like table data (multiple columns)
                    if len(line.split()) >= 3 and any(char.isdigit() for char in line):
                        current_table.append(line)
                    elif current_table:
                        # End of table detected
                        if len(current_table) >= 2:  # At least 2 rows to be a table
                            page_tables += 1
                            table_content = '\n'.join(current_table)
                            table_info = {
                                'type': 'table',
                                'content': table_content,
                                'page': page_num + 1,
                                'table_index': len(tables),
                                'table_data': [{'row': i, 'content': line} for i, line in enumerate(current_table)],
                                'metadata': {
                                    'chunk_type': 'table',
                                    'page_number': page_num + 1,
                                    'table_index': len(tables),
                                    'rows': len(current_table),
                                    'columns': 'unknown',
                                    'column_names': ['row', 'content'],
                                    'extraction_method': 'pattern_matching'
                                }
                            }
                            tables.append(table_info)
                            logger.info(f"Pattern matching: Page {page_num + 1} - Found table {page_tables} with {len(current_table)} rows")
                        current_table = []
                
                # Handle last table on page
                if current_table and len(current_table) >= 2:
                    page_tables += 1
                    table_content = '\n'.join(current_table)
                    table_info = {
                        'type': 'table',
                        'content': table_content,
                        'page': page_num + 1,
                        'table_index': len(tables),
                        'table_data': [{'row': i, 'content': line} for i, line in enumerate(current_table)],
                        'metadata': {
                            'chunk_type': 'table',
                            'page_number': page_num + 1,
                            'table_index': len(tables),
                            'rows': len(current_table),
                            'columns': 'unknown',
                            'column_names': ['row', 'content'],
                            'extraction_method': 'pattern_matching'
                        }
                    }
                    tables.append(table_info)
                    logger.info(f"Pattern matching: Page {page_num + 1} - Found final table {page_tables} with {len(current_table)} rows")
            
            doc.close()
            if tables:
                logger.info(f"Pattern matching: Successfully extracted {len(tables)} tables using fallback method")
            else:
                logger.warning("Pattern matching: No tables found using fallback method")
            return tables
            
        except Exception as e:
            logger.error(f"Pattern matching table extraction failed: {e}")
            return []
    
    def process_pdf(self, pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Process PDF and extract all chunks (text, images, tables)."""
        logger.info(f"=== Starting PDF Processing: {pdf_path} ===")
        
        # Extract different types of content
        logger.info("Step 1: Extracting text chunks...")
        text_chunks = self.extract_text_chunks(pdf_path)
        
        logger.info("Step 2: Extracting images...")
        images = self.extract_images(pdf_path)
        
        logger.info("Step 3: Extracting tables...")
        tables = self.extract_tables(pdf_path)
        
        logger.info(f"=== PDF Processing Complete ===")
        logger.info(f"Summary: {len(text_chunks)} text chunks, {len(images)} images, {len(tables)} tables")
        
        return {
            'text_chunks': text_chunks,
            'images': images,
            'tables': tables
        } 