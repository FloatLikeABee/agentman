"""
Standalone PDF Downloader
Downloads all PDF files linked from a given URL.
"""
import os
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import List

import requests
from bs4 import BeautifulSoup


def download_pdfs_from_url(url: str, download_dir: str = "downloads") -> List[str]:
    """
    Download all PDF files linked from a given URL.
    
    Args:
        url: The URL of the webpage containing PDF links
        download_dir: Directory to save downloaded PDFs (default: "downloads")
    
    Returns:
        List of paths to downloaded PDF files
    """
    # Create download directory if it doesn't exist
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching page: {url}")
    
    # Fetch the webpage
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []
    
    # Parse HTML to find PDF links
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all links that point to PDF files
    pdf_links = []
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        # Check if link ends with .pdf or contains .pdf
        if '.pdf' in href.lower():
            full_url = urljoin(url, href)
            pdf_links.append(full_url)
    
    # Also check for direct PDF links in iframes, embed tags, etc.
    for tag in soup.find_all(['iframe', 'embed', 'object']):
        src = tag.get('src') or tag.get('data')
        if src and '.pdf' in src.lower():
            full_url = urljoin(url, src)
            if full_url not in pdf_links:
                pdf_links.append(full_url)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_pdf_links = []
    for link in pdf_links:
        if link not in seen:
            seen.add(link)
            unique_pdf_links.append(link)
    
    if not unique_pdf_links:
        print("No PDF links found on the page.")
        return []
    
    print(f"Found {len(unique_pdf_links)} PDF link(s)")
    
    # Download each PDF
    downloaded_files = []
    for i, pdf_url in enumerate(unique_pdf_links, 1):
        try:
            print(f"[{i}/{len(unique_pdf_links)}] Downloading: {pdf_url}")
            
            # Get the PDF file
            pdf_response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
            pdf_response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = pdf_response.headers.get('Content-Type', '').lower()
            if 'pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
                # Check first few bytes for PDF magic number
                first_bytes = pdf_response.content[:4]
                if first_bytes != b'%PDF':
                    print(f"  Warning: {pdf_url} doesn't appear to be a PDF, skipping...")
                    continue
            
            # Generate filename from URL
            parsed_url = urlparse(pdf_url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not filename.endswith('.pdf'):
                # Generate a filename from the URL
                domain = parsed_url.netloc.replace('.', '_')
                filename = f"{domain}_{i}.pdf"
            
            # Ensure filename is safe
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            # Save the PDF
            file_path = download_path / filename
            
            # Handle duplicate filenames
            counter = 1
            original_file_path = file_path
            while file_path.exists():
                name_parts = original_file_path.stem, original_file_path.suffix
                file_path = download_path / f"{name_parts[0]}_{counter}{name_parts[1]}"
                counter += 1
            
            with open(file_path, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = file_path.stat().st_size
            print(f"  ✓ Saved: {file_path} ({file_size:,} bytes)")
            downloaded_files.append(str(file_path))
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error downloading {pdf_url}: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error with {pdf_url}: {e}")
    
    print(f"\nDownload complete! {len(downloaded_files)} PDF(s) saved to {download_path}")
    return downloaded_files


def main():
    """Main function for standalone execution"""
    if len(sys.argv) < 2:
        print("Usage: python page_pdf_downloader.py <URL> [download_directory]")
        print("\nExample:")
        print("  python page_pdf_downloader.py https://example.com/pdfs")
        print("  python page_pdf_downloader.py https://example.com/pdfs my_pdfs")
        sys.exit(1)
    
    url = sys.argv[1]
    download_dir = sys.argv[2] if len(sys.argv) > 2 else "downloads"
    
    download_pdfs_from_url(url, download_dir)


if __name__ == "__main__":
    main()
