import os
from bs4 import BeautifulSoup
import re


def find_username_in_html(file_path):
    """
    Scans a single HTML file to find a specific username embedded in a product list.

    Args:
        file_path (str): The path to the HTML file.

    Returns:
        str or None: The found username, or None if not found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'lxml')

        # Find the product table by its class
        product_table = soup.find('table', class_='table1')

        if not product_table:
            return None

        # Find all rows in the table body
        rows = product_table.find('tbody').find_all('tr')

        # Look for the username in the first cell of each row
        for row in rows:
            product_cell = row.find('td')
            if product_cell:
                text_content = product_cell.get_text(strip=True)

                # A flexible regular expression to find a word-like string followed by a colon
                match = re.match(r'(\w+):', text_content)
                if match:
                    # The username is the first captured group
                    username = match.group(1)
                    return username

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None


def scan_folder(folder_path):
    """
    Scans all HTML files in a given folder to find usernames.

    Args:
        folder_path (str): The path to the folder to scan.
    """
    print(f"Scanning folder: {folder_path}\n")

    # Use os.walk() to traverse the directory tree.
    # It yields (dirpath, dirnames, filenames) for each directory.
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an HTML file
            if file.endswith(('.html', '.htm')):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # Call the existing function to find the username
                username = find_username_in_html(file_path)

                if username:
                    print(f"  -> Found username: {username}\n")
                else:
                    print(f"  -> No username found in this file.\n")


if __name__ == "__main__":
    # Get the base directory of the current script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Call the new function to start scanning the folder
    scan_folder(BASE_DIR)