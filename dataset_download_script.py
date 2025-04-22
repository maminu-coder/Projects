import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configuration
HOME_DIR = os.path.expanduser("cic-iot-diad-2024/")  # Changed to local path [[9]][[10]]
BASE_URL = "http://cicresearch.ca/IOTDataset/CIC%20IoT-IDAD%20Dataset%202024/Dataset/Device%20Identification_Anomaly%20Detection%20-%20Packet%20Based%20Features/CIC2023/"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
TIMEOUT = (5, 30)
MAX_WORKERS = 5

#Create session with retry strategy
def setup_session():
    """Create session with retry strategy"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1,
                   status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

#Thread-safe download with progress handling
def download_file(session, url, filename):
    """Thread-safe download with progress handling"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists [[10]]
    if os.path.exists(filename):
        print(f"Already exists: {filename}")
        return
        
    try:
        with session.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        print(f"Downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed: {url} - {str(e)}")

#Recursively process directories with connection reuse
def process_directory(session, url, path):
    """Recursively process directories with connection reuse"""
    try:
        response = session.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
    except requests.exceptions.RequestException as e:
        print(f"Directory access failed: {url} - {str(e)}")
        return

    # Process links
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for link in soup.find_all('a'):
            href = link.get('href', '')
            absolute_url = urljoin(url, href)
            local_path = os.path.join(path, href)

            if href.endswith('.csv'):
                filename = os.path.join(HOME_DIR, local_path)
                futures.append(executor.submit(
                    download_file, session, absolute_url, filename))
                
            elif href.endswith('/') and not href.startswith('..'):
                # Recursively process subdirectories
                futures.append(executor.submit(
                    process_directory, session, absolute_url, local_path))
                
    # Wait for all downloads in this directory
    for future in as_completed(futures):
        future.result()

def main():
    # Validate configuration
    os.makedirs(HOME_DIR, exist_ok=True)  # Create base directory if needed [[10]]
        
    with setup_session() as session:
        session.headers.update(HEADERS)
        process_directory(session, BASE_URL, '')

if __name__ == "__main__":
    main()