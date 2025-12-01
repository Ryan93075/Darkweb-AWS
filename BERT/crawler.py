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
import argparse
from google import genai

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

PROXIES = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050'
}

S3_BUCKET = os.environ.get('S3_BUCKET', 'scraped-data-01')
S3_REGION = os.environ.get('S3_REGION', 'ap-south-1')
UPLOAD_TO_S3 = os.environ.get('UPLOAD_TO_S3', '1') not in ('0', 'false', 'False')

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    max_concurrency=4,
    multipart_chunksize=8 * 1024 * 1024,
    use_threads=True
)

os.makedirs("scraped_data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("scan", exist_ok=True)

visited_links = set()
visited_lock = threading.Lock()

lock = threading.Lock()
crawled_count = 0
crawled_count_lock = threading.Lock()

engines_completed = 0
engines_completed_lock = threading.Lock()

active_crawls = set()
active_crawls_lock = threading.Lock()

assets_downloaded = 0
assets_downloaded_lock = threading.Lock()

done_event = threading.Event()

TOR_HOST = '127.0.0.1'
TOR_SOCKS_PORT = 9050

GEMINI_API_KEY = "AIzaSyCJ0tBXP5zPVwDn4VrbetvCdoFF4iT4lnA"
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

class GeminiLLM:
    def __init__(self, model=None):
        self.model = model or os.environ.get('GEMINI_MODEL', 'gemini-1.5-mini')
        self.max_tokens = int(os.environ.get('GEMINI_MAX_TOKENS', '512'))
        self.temperature = float(os.environ.get('GEMINI_TEMPERATURE', '0.0'))

    def generate(self, prompt, max_tokens=None, temperature=None):
        tok = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature
        try:
            resp = _gemini_client.responses.create(model=self.model, input=prompt, max_output_tokens=tok, temperature=temp)
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text.strip()
            # fallback: try to extract from structured output
            out_items = []
            if hasattr(resp, "output") and resp.output:
                for o in resp.output:
                    if isinstance(o, dict) and "content" in o:
                        for c in o["content"]:
                            if isinstance(c, dict) and "text" in c:
                                out_items.append(c["text"])
            if out_items:
                return "\n".join(out_items).strip()
            return str(resp)
        except Exception as e:
            raise

def get_gemini_llm():
    return GeminiLLM()

def refine_query(llm, user_input):
    system_prompt = (
        "You are a Cybercrime Threat Intelligence Expert. Your task is to refine the provided user query that needs to be sent to darkweb search engines."
        "\nRules:\n1. Analyze the user query and think about how it can be improved to use as search engine query"
        "\n2. Refine the user query by adding or removing words so that it returns the best result from dark web search engines"
        "\n3. Don't use any logical operators (AND, OR, etc.)\n4. Output just the user query and nothing else\n\nINPUT:\n"
    )
    prompt = system_prompt + user_input
    out = llm.generate(prompt, max_tokens=128)
    return out.strip()

def _generate_final_string(results, truncate=False):
    if truncate:
        max_title_length = 30
        max_link_length = 0
    final_str = []
    for i, res in enumerate(results):
        truncated_link = re.sub(r"(?<=\.onion).*", "", res.get('link', ''))
        title = re.sub(r"[^0-9a-zA-Z\-\.]", " ", res.get('title', ''))
        if truncated_link == "" and title == "":
            continue
        if truncate:
            title = (title[:max_title_length] + "...") if len(title) > max_title_length else title
            truncated_link = (truncated_link[:max_link_length] + "...") if len(truncated_link) > max_link_length else truncated_link
        final_str.append(f"{i+1}. {truncated_link} - {title}")
    return "\n".join(s for s in final_str)

def filter_results(llm, query, results):
    if not results:
        return []
    system_prompt = (
        "You are a Cybercrime Threat Intelligence Expert. You are given a dark web search query and a list of search results in the form of index, link and title."
        "\nYour task is select the Top 20 relevant results that best match the search query for user to investigate more.\nRule:\n1. Output ONLY atmost top 20 indices (comma-separated list) no more than that that best match the input query\n\nSearch Query: "
    )
    final_str = _generate_final_string(results)
    prompt = system_prompt + query + "\nSearch Results:\n" + final_str
    try:
        result_indices = llm.generate(prompt, max_tokens=256)
    except Exception:
        final_str = _generate_final_string(results, truncate=True)
        prompt = system_prompt + query + "\nSearch Results:\n" + final_str
        result_indices = llm.generate(prompt, max_tokens=256)
    parsed_indices = []
    for match in re.findall(r"\d+", result_indices):
        try:
            idx = int(match)
            if 1 <= idx <= len(results):
                parsed_indices.append(idx)
        except ValueError:
            continue
    seen = set()
    parsed_indices = [i for i in parsed_indices if not (i in seen or seen.add(i))]
    if not parsed_indices:
        parsed_indices = list(range(1, min(len(results), 20) + 1))
    top_results = [results[i - 1] for i in parsed_indices[:20]]
    return top_results

def get_s3_client():
    session = boto3.session.Session()
    return session.client('s3', region_name=S3_REGION)

def upload_file_to_s3(s3_client, local_path, bucket, s3_key):
    try:
        content_type, _ = mimetypes.guess_type(local_path)
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        extra_args['ACL'] = 'private'
        s3_client.upload_file(local_path, bucket, s3_key, ExtraArgs=extra_args, Config=TRANSFER_CONFIG)
        print(f"\n⬆️ Uploaded {local_path} -> s3://{bucket}/{s3_key}")
    except botocore.exceptions.ClientError as e:
        print(f"\n⚠️ Failed to upload {local_path} -> s3://{bucket}/{s3_key} : {e}")
    except Exception as e:
        print(f"\n⚠️ Unexpected error uploading {local_path}: {e}")

def upload_folder_to_s3(local_folder, bucket, s3_prefix=""):
    if not UPLOAD_TO_S3:
        print("\n🛈 S3 uploads disabled by UPLOAD_TO_S3 env var.")
        return
    if not os.path.isdir(local_folder):
        print(f"\n⚠️ Local folder does not exist, skipping upload: {local_folder}")
        return
    s3_client = get_s3_client()
    if s3_prefix and not s3_prefix.endswith('/'):
        s3_prefix = s3_prefix + '/'
    files_to_upload = []
    for root, dirs, files in os.walk(local_folder):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_folder)
            s3_key = (s3_prefix + rel_path).replace(os.path.sep, '/')
            files_to_upload.append((local_path, s3_key))
    if not files_to_upload:
        print(f"\n🛈 Nothing to upload from {local_folder}")
        return
    print(f"\n🚚 Uploading {len(files_to_upload)} files from {local_folder} to s3://{bucket}/{s3_prefix} ...")
    with ThreadPoolExecutor(max_workers=4) as exec:
        futures = [exec.submit(upload_file_to_s3, s3_client, lp, bucket, sk) for lp, sk in files_to_upload]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                pass
    print(f"✅ Finished uploading {local_folder} -> s3://{bucket}/{s3_prefix}")

def is_socks_port_open(host='127.0.0.1', port=9050, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def wait_for_tor_ready(timeout=30, interval=1, require_success=False):
    start = time.time()
    saw_port = False
    last_msg = 0
    while time.time() - start < timeout:
        if is_socks_port_open(TOR_HOST, TOR_SOCKS_PORT, timeout=1.0):
            saw_port = True
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

def download_content(url, folder_name):
    global assets_downloaded
    try:
        response = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=12)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            extensions = {"text/html": "html"}
            ctype = content_type.split(';')[0].strip().lower()
            ext = extensions.get(ctype)
            if not ext:
                guessed, _ = mimetypes.guess_type(url)
                if guessed:
                    ext = guessed.split('/')[-1]
            if not ext:
                return
            filename = os.path.join(folder_name, f"file_{int(time.time() * 1000)}.{ext}")
            os.makedirs(folder_name, exist_ok=True)
            with open(filename, 'wb') as f:
                f.write(response.content)
            with assets_downloaded_lock:
                assets_downloaded += 1
    except Exception:
        pass

def download_assets_parallel(soup, base_url, folder_name):
    tags = soup.find_all(["img", "script", "link", "a"])
    urls = []
    for tag in tags:
        attr = "href" if tag.name in ["link", "a"] else "src"
        val = tag.get(attr)
        if val:
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
    links = set()
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a['href']
            if ".onion" in href:
                full = urljoin(base_url or "", href)
                links.add(full)
        found = re.findall(r'([A-Za-z0-9_-]{16,60}\.onion(?:/[^\s"\'"<>]*)?)', html)
        for f in found:
            if f.startswith('http://') or f.startswith('https://'):
                links.add(f)
            else:
                links.add('http://' + f)
    except Exception:
        pass
    return list(links)

def normalize_onion_links(links, max_results=30, engine_netloc=None):
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
            if engine_netloc and engine_netloc.lower().strip() in netloc:
                continue
            lower_path = parsed.path.lower()
            if any(lower_path.endswith(ext) for ext in asset_exts):
                continue
            if any(sub in lower_path for sub in blacklist_substrings):
                continue
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

def crawl_single_site(url):
    global crawled_count
    with visited_lock:
        if url in visited_links:
            return
        visited_links.add(url)
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
    if status == 'ok' and response is not None:
        new_links = extract_onion_links_from_html(response.text, base_url=url)
        normalized = normalize_onion_links(new_links, max_results=50)
        with visited_lock:
            for link in normalized:
                if link not in visited_links:
                    visited_links.add(link)
    with crawled_count_lock:
        crawled_count += 1
    with active_crawls_lock:
        if url in active_crawls:
            active_crawls.remove(url)
    print(f"\r✅ Crawled: {crawled_count} | Last: {domain} | Status: {status[:40]}{' ' * 10}", end='', flush=True)
    if UPLOAD_TO_S3:
        try:
            s3_prefix = f"scan/{domain}"
            upload_folder_to_s3(folder_name, S3_BUCKET, s3_prefix)
        except Exception as e:
            print(f"\n⚠️ Failed to upload folder for {domain}: {e}")

def try_build_and_fetch(engine, keyword, try_params=('q','query','search','s')):
    proxies = None
    if engine.get('type') == 'onion':
        if PROXIES is None:
            return None
        proxies = PROXIES
    engine_base = engine['url']
    try:
        home = requests.get(engine_base, headers=HEADERS, proxies=proxies, timeout=12)
        if home.status_code == 200 and home.text:
            soup = BeautifulSoup(home.text, 'html.parser')
            forms = soup.find_all('form')
            for form in forms:
                method = (form.get('method') or 'get').lower()
                action = form.get('action') or engine_base
                action_url = urljoin(engine_base, action)
                input_candidates = []
                for inp in form.find_all('input'):
                    name = inp.get('name')
                    if not name:
                        continue
                    itype = (inp.get('type') or 'text').lower()
                    if itype in ('text', 'search') or any(k in name.lower() for k in ('q','query','search','s')):
                        input_candidates.append(name)
                if not input_candidates:
                    for inp in form.find_all(['textarea']):
                        name = inp.get('name')
                        if name:
                            input_candidates.append(name)
                if input_candidates:
                    params = {input_candidates[0]: keyword}
                    try:
                        if method == 'post':
                            r = requests.post(action_url, data=params, headers=HEADERS, proxies=proxies, timeout=20)
                        else:
                            r = requests.get(action_url, params=params, headers=HEADERS, proxies=proxies, timeout=20)
                        if r.status_code == 200 and r.text and len(r.text) > 50:
                            return r.text
                    except Exception:
                        pass
    except Exception:
        pass
    for p in try_params:
        try:
            r = requests.get(engine_base, params={p: keyword}, headers=HEADERS, proxies=proxies, timeout=16)
            if r.status_code == 200 and r.text and len(r.text) > 50:
                return r.text
        except Exception:
            continue
    try:
        fallback_url = urljoin(engine_base, requests.utils.requote_uri(keyword))
        r = requests.get(fallback_url, headers=HEADERS, proxies=proxies, timeout=16)
        if r.status_code == 200 and r.text and len(r.text) > 50:
            return r.text
    except Exception:
        pass
    return None

def search_on_search_engine(keyword, engine, max_results=30):
    global engines_completed
    html = try_build_and_fetch(engine, keyword)
    found_links = []
    engine_netloc = None
    try:
        engine_netloc = urlparse(engine['url']).netloc
    except Exception:
        engine_netloc = None
    if not html:
        print(f"\n⚠️  {engine['name']} did not return valid results (captcha/offline/different API).")
    else:
        raw_links = extract_onion_links_from_html(html, base_url=engine['url'])
        found_links = normalize_onion_links(raw_links, max_results=max_results, engine_netloc=engine_netloc)
        print(f"\n🔎 {engine['name']} -> found {len(found_links)} onion root links (sample: {found_links[:3]})")
    with engines_completed_lock:
        engines_completed += 1
    return found_links

def ahmia_clearnet_search(keyword, max_results=200):
    url = "https://ahmia.fi/search/"
    try:
        r = requests.get(url, params={'q': keyword}, headers=HEADERS, timeout=20)
        if r.status_code == 200 and r.text:
            raw = extract_onion_links_from_html(r.text, base_url=url)
            normalized = normalize_onion_links(raw, max_results=max_results)
            print(f"\n🔎 Ahmia (clearnet) -> found {len(normalized)} onion root links (sample: {normalized[:3]})")
            return normalized
    except Exception as e:
        print(f"\n⚠️ Ahmia clearnet search failed: {e}")
    return []

def status_printer(total_engines, candidate_sites_ref):
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

def start_crawl():
    keyword = input("🔍 Enter keyword to search for on dark-web search engines: ").strip()
    if not keyword:
        print("No keyword entered, aborting.")
        return
    try:
        llm = get_gemini_llm()
        refined = refine_query(llm, keyword)
        if refined:
            print(f"Refined query: {refined}")
            search_keyword = refined
        else:
            search_keyword = keyword
    except Exception as e:
        print(f"LLM refine step failed: {e} \nProceeding with original keyword")
        search_keyword = keyword
    search_engines = [
        {'name': 'Torch', 'url': 'http://torchdeedp3i2jigzjdmfpn5ttjhthh5wbmda2rr3jvqjg5p77c54dqd.onion/', 'type': 'onion'},
        {'name': 'Ahmia (clearnet)', 'url': 'https://ahmia.fi/search/', 'type': 'clearnet'},
    ]
    total_engines = len(search_engines)
    candidate_sites = []
    stat_thread = threading.Thread(target=status_printer, args=(total_engines, candidate_sites), daemon=True)
    stat_thread.start()
    with ThreadPoolExecutor(max_workers=min(10, total_engines)) as s_exec:
        future_to_engine = {}
        for engine in search_engines:
            if engine.get('type') == 'onion' and PROXIES is None:
                print(f"⚠️ Skipping onion engine {engine['name']} because PROXIES is not configured.")
                with engines_completed_lock:
                    engines_completed += 1
                continue
            future = s_exec.submit(search_on_search_engine, search_keyword, engine, 50)
            future_to_engine[future] = engine
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
        print("\n🛈 No results from primary engines — trying Ahmia (clearnet) fallback...")
        ahmia_links = ahmia_clearnet_search(search_keyword, max_results=200)
        for f in ahmia_links:
            if f not in candidate_sites:
                candidate_sites.append(f)
    if not candidate_sites:
        print("\n❌ No results found on configured/available search engines (including Ahmia fallback).")
        done_event.set()
        return
    try:
        llm = get_gemini_llm()
        results_for_filter = [{'link': u, 'title': u} for u in candidate_sites]
        top = filter_results(llm, search_keyword, results_for_filter)
        if top:
            candidate_sites = [r['link'] for r in top]
            print(f"\n🧰 LLM filtered to {len(candidate_sites)} candidate sites")
    except Exception as e:
        print(f"LLM filtering step failed: {e} \nProceeding with unfiltered candidate list")
    print(f"\n\n🚀 Starting crawl on {len(candidate_sites)} discovered sites (top {min(50, len(candidate_sites))} shown):")
    for s in candidate_sites[:50]:
        print("  -", s)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(crawl_single_site, link) for link in candidate_sites]
        for _ in as_completed(futures):
            pass
    done_event.set()
    time.sleep(1.0)
    if UPLOAD_TO_S3:
        try:
            upload_folder_to_s3("scan", S3_BUCKET, "scan")
        except Exception as e:
            print(f"\n⚠️ Final upload of scan/ failed: {e}")

def parse_args_and_prepare():
    parser = argparse.ArgumentParser(description="Dark-web crawler (uses local Tor SOCKS5 at 127.0.0.1:9050)")
    parser.add_argument("--socks", help="SOCKS host:port (default 127.0.0.1:9050)", default=f"{TOR_HOST}:{TOR_SOCKS_PORT}")
    parser.add_argument("--require-tor-check", action="store_true", help="Require check.torproject.org success before proceeding")
    args = parser.parse_args()
    global PROXIES
    if args.socks and args.socks != f"{TOR_HOST}:{TOR_SOCKS_PORT}":
        host, port = args.socks.split(":")
        PROXIES = {
            'http': f'socks5h://{host}:{port}',
            'https': f'socks5h://{host}:{port}'
        }
    return args

if __name__ == '__main__':
    args = parse_args_and_prepare()
    if is_socks_port_open(TOR_HOST, TOR_SOCKS_PORT, timeout=0.7):
        print(f"🟢 Found existing Tor SOCKS listener on {TOR_HOST}:{TOR_SOCKS_PORT} — using system tor or another instance.")
        tor_ready = wait_for_tor_ready(timeout=30, interval=1, require_success=args.require_tor_check)
        if not tor_ready:
            print("\n⚠️ Tor SOCKS port present but failed readiness check. Proceeding anyway for onion-only crawling.")
        else:
            print("✅ Tor OK — requests will be routed through SOCKS5 at 127.0.0.1:9050")
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
