"""HTTP configuration shared by the portable cache experiments; no key persistence."""
import json, os, urllib.parse, urllib.request
_ENDPOINT = None
_HEADERS = None

def configure(base_url, api_key_env):
    global _ENDPOINT, _HEADERS
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use an HTTP(S) base URL without credentials, query, or fragment')
    _ENDPOINT = base_url.rstrip('/') + '/chat/completions'
    key = os.environ.get(api_key_env, '')
    _HEADERS = {'Content-Type': 'application/json'}
    if key: _HEADERS['Authorization'] = 'Bearer ' + key

def request(payload):
    if _ENDPOINT is None: raise RuntimeError('Configure HTTP first')
    return urllib.request.Request(_ENDPOINT, data=json.dumps(payload).encode(), headers=_HEADERS)
