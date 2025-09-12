import requests

proxies = {
    "http": "socks5h://127.0.0.1:9150",
    "https": "socks5h://127.0.0.1:9150"
}

r = requests.get("http://xmh57jrknzkhv6y3ls3ubitzfqnkrwxhopf5aygthi7d6rplyvk3noyd.onion", proxies=proxies, timeout=30)
print(r.status_code)
print(r.text[:500])
