#!/usr/bin/env python3
import os
import requests
import time
import re
import mimetypes
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import boto3
import botocore
from boto3.s3.transfer import TransferConfig
import socket
import sys

# ---------------- CONFIG ----------------
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# default proxies point to local tor socks5 (will be used only if a Tor listener exists)
PROXIES = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050'
}
S3_BUCKET = os.environ.get('S3_BUCKET', 'scraped-data-01')  # override via env if desired
S3_REGION = os.environ.get('S3_REGION', 'ap-south-1')
UPLOAD_TO_S3 = os.environ.get('UPLOAD_TO_S3', '1') not in ('0', 'false', 'False')  # set to 0/false to disable

# Transfer config for efficient uploads (adjust as needed)
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,  # 8MB
    max_concurrency=4,
    multipart_chunksize=8 * 1024 * 1024,
    use_threads=True
)

# ---------------- FS setup ----------------
os.makedirs("scraped_data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("scan", exist_ok=True)

# ---------- shared state & locks ----------
visited_links = set()
visited_lock = threading.Lock()

lock = threading.Lock()  # generic lock used in several places
crawled_count = 0
crawled_count_lock = threading.Lock()

engines_completed = 0
engines_completed_lock = threading.Lock()

active_crawls = set()
active_crawls_lock = threading.Lock()

assets_downloaded = 0
assets_downloaded_lock = threading.Lock()

done_event = threading.Event()  # signals status printer to finish

# ---------- S3 helper ----------
def get_s3_client():
    """
    Create an S3 client. Credentials are picked up from environment, IAM role, or ~/.aws/credentials.
    """
    session = boto3.session.Session()
    return session.client('s3', region_name=S3_REGION)

def upload_file_to_s3(s3_client, local_path, bucket, s3_key):
    try:
        content_type, _ = mimetypes.guess_type(local_path)
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        # Keep uploads private
        extra_args['ACL'] = 'private'
        s3_client.upload_file(local_path, bucket, s3_key, ExtraArgs=extra_args, Config=TRANSFER_CONFIG)
        print(f"\n⬆️ Uploaded {local_path} -> s3://{bucket}/{s3_key}")
    except botocore.exceptions.ClientError as e:
        print(f"\n⚠️ Failed to upload {local_path} -> s3://{bucket}/{s3_key} : {e}")
    except Exception as e:
        print(f"\n⚠️ Unexpected error uploading {local_path}: {e}")

def upload_folder_to_s3(local_folder, bucket, s3_prefix=""):
    """
    Upload entire folder recursively to S3 under s3_prefix (prefix may be empty or end without slash).
    Example: upload_folder_to_s3("scan/example", "scraped-data-01", "scan/example/")
    """
    if not UPLOAD_TO_S3:
        print("\n🛈 S3 uploads disabled by UPLOAD_TO_S3 env var.")
        return

    if not os.path.isdir(local_folder):
        print(f"\n⚠️ Local folder does not exist, skipping upload: {local_folder}")
        return

    s3_client = get_s3_client()

    # Ensure prefix ends with slash if non-empty
    if s3_prefix and not s3_prefix.endswith('/'):
        s3_prefix = s3_prefix + '/'

    files_to_upload = []
    for root, dirs, files in os.walk(local_folder):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_folder)
            # Build S3 key using prefix + rel_path (use forward slashes)
            s3_key = (s3_prefix + rel_path).replace(os.path.sep, '/')
            files_to_upload.append((local_path, s3_key))

    if not files_to_upload:
        print(f"\n🛈 Nothing to upload from {local_folder}")
        return

    print(f"\n🚚 Uploading {len(files_to_upload)} files from {local_folder} to s3://{bucket}/{s3_prefix} ...")
    # Upload in parallel to speed up
    with ThreadPoolExecutor(max_workers=4) as exec:
        futures = [exec.submit(upload_file_to_s3, s3_client, lp, bucket, sk) for lp, sk in files_to_upload]
        for fut in as_completed(futures):
            # just iterate to surface exceptions
            try:
                fut.result()
            except Exception:
                pass
    print(f"✅ Finished uploading {local_folder} -> s3://{bucket}/{s3_prefix}")

# ---------- Tor helpers ----------
TOR_HOST = '127.0.0.1'
TOR_SOCKS_PORT = 9050

def is_socks_port_open(host='127.0.0.1', port=9050, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def wait_for_tor_ready(timeout=30, interval=1, require_success=False):
    """
    Wait until Tor's SOCKS5 port is accepting connections.
    If require_success were True we'd also try checking check.torproject.org through the proxy,
    but for onion-only crawling we accept a simple open socket as "ready".
    """
    start = time.time()
    saw_port = False
    last_msg = 0
    while time.time() - start < timeout:
        if is_socks_port_open(TOR_HOST, TOR_SOCKS_PORT, timeout=1.0):
            saw_port = True
            # If user requested an extra check, do it (but often check.torproject.org fails from EC2)
            if require_success:
                try:
                    r = requests.get("https://check.torproject.org/", proxies=PROXIES, timeout=6)
                    if r.status_code == 200 and "Congratulations" in r.text:
                        print("✅ Tor is ready (check.torproject.org ok)")
                        return True
                    else:
                        print("⚠️ Tor port open but check.torproject.org didn't confirm. Proceeding (onion-only ok).")
                        return True
                except Exception:
                    print("⚠️ Tor port open but check through proxy failed. Proceeding (onion-only ok).")
                    return True
            else:
                print("✅ Tor SOCKS port open (onion-only crawling acceptable).")
                return True
        else:
            now = time.time()
            if now - last_msg > 5:
                print(f"Waiting for Tor SOCKS port to open on {TOR_HOST}:{TOR_SOCKS_PORT}...")
                last_msg = now
        time.sleep(interval)
    if not saw_port:
        print(f"\n⚠️ Timed out waiting for Tor SOCKS port to open on {TOR_HOST}:{TOR_SOCKS_PORT}.")
    else:
        print("\n⚠️ Timed out waiting for Tor to fully bootstrap (check.torproject.org didn't report success).")
    return False

# ---------- downloader/crawler helpers ----------
def download_content(url, folder_name):
    global assets_downloaded
    try:
        response = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=12)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            extensions = {
                "text/html": "html"
            }
            # choose extension by content-type prefix
            ctype = content_type.split(';')[0].strip().lower()
            ext = extensions.get(ctype)
            if not ext:
                # fallback guess from URL path
                guessed, _ = mimetypes.guess_type(url)
                if guessed:
                    ext = guessed.split('/')[-1]
            if not ext:
                return
            filename = os.path.join(folder_name, f"file_{int(time.time() * 1000)}.{ext}")
            # ensure folder exists
            os.makedirs(folder_name, exist_ok=True)
            with open(filename, 'wb') as f:
                f.write(response.content)
            # increment assets counter
            with assets_downloaded_lock:
                assets_downloaded += 1
    except Exception:
        # silent fail — don't crash crawler for optional assets
        pass

def download_assets_parallel(soup, base_url, folder_name):
    tags = soup.find_all(["img", "script", "link", "a"])
    urls = []
    for tag in tags:
        attr = "href" if tag.name in ["link", "a"] else "src"
        val = tag.get(attr)
        if val:
            # join relative urls
            try:
                full = urljoin(base_url, val)
                urls.append(full)
            except Exception:
                continue

    if not urls:
        return

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(download_content, u, folder_name) for u in urls]
        for _ in as_completed(futures):
            pass

def extract_onion_links_from_html(html, base_url=None):
    """
    Extract onion links, both with and without scheme.
    Returns normalized URLs (adds http:// if scheme missing).
    """
    links = set()
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a['href']
            if ".onion" in href:
                # normalize relative links
                full = urljoin(base_url or "", href)
                links.add(full)

        # find raw .onion occurrences (with optional path), including bare domains
        # matches: abcdefghijklmnop.onion or abcdefghijklmnop.onion/some/path
        found = re.findall(r'([A-Za-z0-9_-]{16,56}\.onion(?:/[^\s"\'<>]*)?)', html)
        for f in found:
            if f.startswith('http://') or f.startswith('https://'):
                links.add(f)
            else:
                links.add('http://' + f)  # prefer http scheme for onion
    except Exception:
        pass
    return list(links)

def normalize_onion_links(links, max_results=30, engine_netloc=None):
    """
    Normalize the raw links found on a page:
    - keep only .onion links
    - drop obvious assets (images, css, js, fonts, etc.)
    - return domain-level roots (scheme + netloc + '/')
    - optionally skip links that match the search engine domain (to avoid crawling engine itself)
    - preserve ordering, limit to max_results
    """
    seen = set()
    out = []
    asset_exts = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.css', '.js', '.ico', '.woff', '.woff2', '.ttf', '.eot')
    blacklist_substrings = ('/banners/', '/banner', '/images/', '/img/', '/ads/', '/advert', '/api/', '/static/', '/cdn/')
    for raw in links:
        try:
            candidate = raw if raw.startswith('http') else ('http://' + raw)
            parsed = urlparse(candidate)
            netloc = parsed.netloc.lower()
            if '.onion' not in netloc:
                continue
            # skip if it's the engine domain itself
            if engine_netloc and engine_netloc.lower().strip() in netloc:
                continue
            # drop asset-like URLs by extension in path
            lower_path = parsed.path.lower()
            if any(lower_path.endswith(ext) for ext in asset_exts):
                continue
            # drop common blacklisted path pieces
            if any(sub in lower_path for sub in blacklist_substrings):
                continue
            # derive root
            scheme = parsed.scheme or 'http'
            root = f"{scheme}://{parsed.netloc.rstrip('/')}/"
            if root not in seen:
                seen.add(root)
                out.append(root)
            if len(out) >= max_results:
                break
        except Exception:
            continue
    return out

# ---------- crawling a single site ----------
def crawl_single_site(url):
    """
    Crawl a single site: save index.html, download assets (images/js/css), discover .onion links.
    Upload the site's folder to S3 after crawling.
    """
    global crawled_count
    # dedupe & mark as visited
    with visited_lock:
        if url in visited_links:
            return
        visited_links.add(url)

    # mark active crawl
    with active_crawls_lock:
        active_crawls.add(url)

    status = 'ok'
    response = None
    try:
        response = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=25)
        response.raise_for_status()
    except Exception as e:
        status = f'error:{str(e)[:120]}'

    domain = urlparse(url).netloc.replace('.onion', '')
    folder_name = os.path.join("scan", domain)
    os.makedirs(folder_name, exist_ok=True)

    if status == 'ok' and response is not None:
        try:
            index_path = os.path.join(folder_name, "index.html")
            with open(index_path, 'wb') as f:
                f.write(response.content)
            soup = BeautifulSoup(response.content, 'html.parser')
            download_assets_parallel(soup, url, folder_name)
        except Exception:
            pass

    # discover additional onion links on the page (bounded)
    if status == 'ok' and response is not None:
        new_links = extract_onion_links_from_html(response.text, base_url=url)
        # normalize new links to domain roots and add them to visited (but don't crawl them immediately here)
        normalized = normalize_onion_links(new_links, max_results=50)
        with visited_lock:
            for link in normalized:
                if link not in visited_links:
                    visited_links.add(link)

    with crawled_count_lock:
        crawled_count += 1

    # unmark active crawl
    with active_crawls_lock:
        if url in active_crawls:
            active_crawls.remove(url)

    # print short progress message
    print(f"\r✅ Crawled: {crawled_count} | Last: {domain} | Status: {status[:40]}{' ' * 10}", end='', flush=True)

    # Upload this site's folder to S3 (best-effort)
    if UPLOAD_TO_S3:
        try:
            # prefix under bucket: scan/<domain>/
            s3_prefix = f"scan/{domain}"
            upload_folder_to_s3(folder_name, S3_BUCKET, s3_prefix)
        except Exception as e:
            print(f"\n⚠️ Failed to upload folder for {domain}: {e}")

# ---------- search engine querying (parallel) ----------
def try_build_and_fetch(engine_base, keyword, try_params=('q','query','search','s')):
    """
    1) Try to auto-discover a search <form> on the engine homepage and submit it using the form's input name.
    2) If no usable form found, fall back to trying common GET params.
    3) Final fallback: try appending the keyword to the path.
    Returns HTML text or None.
    """
    # 1) Fetch homepage, try to discover a form
    try:
        home = requests.get(engine_base, headers=HEADERS, proxies=PROXIES, timeout=12)
        if home.status_code == 200 and home.text:
            soup = BeautifulSoup(home.text, 'html.parser')
            forms = soup.find_all('form')
            for form in forms:
                method = (form.get('method') or 'get').lower()
                action = form.get('action') or engine_base
                action_url = urljoin(engine_base, action)

                # find candidate input names (text/search inputs or names containing q/search/query/s)
                input_candidates = []
                for inp in form.find_all('input'):
                    name = inp.get('name')
                    if not name:
                        continue
                    itype = (inp.get('type') or 'text').lower()
                    if itype in ('text', 'search') or any(k in name.lower() for k in ('q','query','search','s')):
                        input_candidates.append(name)

                if not input_candidates:
                    # also consider <textarea>
                    for inp in form.find_all(['textarea']):
                        name = inp.get('name')
                        if name:
                            input_candidates.append(name)

                if input_candidates:
                    params = {input_candidates[0]: keyword}
                    try:
                        if method == 'post':
                            r = requests.post(action_url, data=params, headers=HEADERS, proxies=PROXIES, timeout=20)
                        else:
                            r = requests.get(action_url, params=params, headers=HEADERS, proxies=PROXIES, timeout=20)
                        if r.status_code == 200 and r.text and len(r.text) > 50:
                            return r.text
                    except Exception:
                        # if submitting form fails, continue to other forms or fallback
                        pass
    except Exception:
        pass

    # 2) If form discovery didn't work: try common GET params
    for p in try_params:
        try:
            r = requests.get(engine_base, params={p: keyword}, headers=HEADERS, proxies=PROXIES, timeout=16)
            if r.status_code == 200 and r.text and len(r.text) > 50:
                return r.text
        except Exception:
            continue

    # 3) Final fallback: append keyword to path (some engines use path-based search)
    try:
        fallback_url = urljoin(engine_base, requests.utils.requote_uri(keyword))
        r = requests.get(fallback_url, headers=HEADERS, proxies=PROXIES, timeout=16)
        if r.status_code == 200 and r.text and len(r.text) > 50:
            return r.text
    except Exception:
        pass

    return None

def search_on_search_engine(keyword, engine, max_results=30):
    """
    Query a single engine and extract onion roots.
    """
    global engines_completed
    html = try_build_and_fetch(engine['url'], keyword)
    found_links = []
    engine_netloc = None
    try:
        engine_netloc = urlparse(engine['url']).netloc
    except Exception:
        engine_netloc = None

    if not html:
        print(f"\n⚠️  {engine['name']} did not return valid results (captcha/offline/different API).")
    else:
        # extract raw onion occurrences
        raw_links = extract_onion_links_from_html(html, base_url=engine['url'])
        # normalize to domain roots & filter assets and engine domain itself
        found_links = normalize_onion_links(raw_links, max_results=max_results, engine_netloc=engine_netloc)
        print(f"\n🔎 {engine['name']} -> found {len(found_links)} onion root links (sample: {found_links[:3]})")

    with engines_completed_lock:
        engines_completed += 1

    return found_links

# ---------- status printer (background) ----------
def status_printer(total_engines, candidate_sites_ref):
    """
    Prints a single-line status every 1.5s until done_event is set.
    """
    while not done_event.is_set():
        with engines_completed_lock:
            done = engines_completed
        with visited_lock:
            visited = len(visited_links)
        with crawled_count_lock:
            crawled = crawled_count
        with active_crawls_lock:
            active = len(active_crawls)
        with assets_downloaded_lock:
            assets = assets_downloaded
        cand = len(candidate_sites_ref)
        print(f"\r[Engines done: {done}/{total_engines}] Candidates: {cand} | Visited: {visited} | Active: {active} | Assets: {assets} | Crawled: {crawled}    ", end='', flush=True)
        time.sleep(1.5)

    # final print after completion
    with engines_completed_lock:
        done = engines_completed
    with visited_lock:
        visited = len(visited_links)
    with crawled_count_lock:
        crawled = crawled_count
    with active_crawls_lock:
        active = len(active_crawls)
    with assets_downloaded_lock:
        assets = assets_downloaded
    print(f"\n[Finished] Engines done: {done}/{len(candidate_sites_ref)} | Candidates: {len(candidate_sites_ref)} | Visited: {visited} | Active: {active} | Assets: {assets} | Crawled: {crawled}")

# ---------- orchestrator (parallel search across engines) ----------
def start_crawl():
    keyword = input("🔍 Enter keyword to search for on dark-web search engines: ").strip()
    if not keyword:
        print("No keyword entered, aborting.")
        return

    # full list of search engines you provided (onion endpoints)
    search_engines = [
        {'name': 'Torch', 'url': 'http://torchdeedp3i2jigzjdmfpn5ttjhthh5wbmda2rr3jvqjg5p77c54dqd.onion/'},
        {'name': 'TorLand', 'url': 'http://torlgu6zhhtwe73fdu76uiswgnkfvukqfujofxjfo7vzoht2rndyhxyd.onion/'},
        {'name': 'Venus', 'url': 'http://venusoseaqnafjvzfmrcpcq6g47rhd7sa6nmzvaa4bj5rp6nm5jl7gad.onion/'},
    ]

    total_engines = len(search_engines)
    candidate_sites = []

    # start status printer thread
    stat_thread = threading.Thread(target=status_printer, args=(total_engines, candidate_sites), daemon=True)
    stat_thread.start()

    # Query all engines in parallel
    with ThreadPoolExecutor(max_workers=min(10, total_engines)) as s_exec:
        future_to_engine = {s_exec.submit(search_on_search_engine, keyword, engine, 25): engine for engine in search_engines}
        for fut in as_completed(future_to_engine):
            engine = future_to_engine[fut]
            try:
                found = fut.result()
                for f in found:
                    if f not in candidate_sites:
                        candidate_sites.append(f)
            except Exception as e:
                print(f"Error searching {engine['name']}: {e}")

    if not candidate_sites:
        print("❌ No results found on configured search engines.")
        # signal done and exit status printer
        done_event.set()
        return

    print(f"\n\n🚀 Starting crawl on {len(candidate_sites)} discovered sites (top {min(10, len(candidate_sites))} shown):")
    for s in candidate_sites[:10]:
        print("  -", s)

    # crawl each discovered site (bounded parallelism)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(crawl_single_site, link) for link in candidate_sites]
        for _ in as_completed(futures):
            pass

    # signal done to status printer and allow it to print final stats
    done_event.set()
    # short wait to allow final status print
    time.sleep(1.0)

    # Final upload of the entire scan folder (best-effort)
    if UPLOAD_TO_S3:
        try:
            upload_folder_to_s3("scan", S3_BUCKET, "scan")
        except Exception as e:
            print(f"\n⚠️ Final upload of scan/ failed: {e}")

if __name__ == '__main__':
    # This script will NOT attempt to start Tor. It only connects to an existing Tor SOCKS5 listener.
    if is_socks_port_open(TOR_HOST, TOR_SOCKS_PORT, timeout=0.7):
        print(f"🟢 Found existing Tor SOCKS listener on {TOR_HOST}:{TOR_SOCKS_PORT} — using system tor or another instance.")
        tor_ready = wait_for_tor_ready(timeout=30, interval=1, require_success=False)
        if not tor_ready:
            print("\n⚠️ Tor SOCKS port present but failed readiness check. Proceeding anyway for onion-only crawling.")
        else:
            print("✅ Tor OK — requests will be routed through SOCKS5 at 127.0.0.1:9050")
        # sanity-check S3 connectivity (best-effort; non-fatal)
        if UPLOAD_TO_S3:
            try:
                client = get_s3_client()
                client.head_bucket(Bucket=S3_BUCKET)
                print(f"✅ S3 bucket '{S3_BUCKET}' reachable in region '{S3_REGION}'.")
            except botocore.exceptions.ClientError as e:
                print(f"⚠️ Warning: S3 head_bucket failed for '{S3_BUCKET}': {e}. Continuing, uploads may fail.")
            except Exception:
                print(f"⚠️ Could not verify S3 bucket '{S3_BUCKET}'. Continuing, uploads may fail.")
        start_crawl()
    else:
        print(f"❌ No Tor SOCKS listener found at {TOR_HOST}:{TOR_SOCKS_PORT}. Please start Tor (systemd, docker, or manually) and re-run the script.")
        sys.exit(1)