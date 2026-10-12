"""Loopback-only authentication adapter for the unchanged verifier HTTP API."""

import http.server
import threading
import urllib.error
import urllib.request


def start_proxy(url, token, port=18081):
    if not url.startswith("https://") or not token:
        raise ValueError("private verifier requires HTTPS and a Connect Token")

    class Handler(http.server.BaseHTTPRequestHandler):
        def proxy(self):
            route = self.path.split("?", 1)[0]
            if route != "/health" and not route.startswith(("/verify/full/", "/verify/anchored/")):
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if size > 8 * 1024 * 1024:
                self.send_error(413)
                return
            request = urllib.request.Request(
                url.rstrip("/") + self.path,
                data=self.rfile.read(size) if self.command == "POST" else None,
                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
                method=self.command,
            )
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    payload, code = response.read(), response.status
            except urllib.error.HTTPError as error:
                payload, code = error.read(), error.code
            except (urllib.error.URLError, OSError, TimeoutError):
                self.send_error(502, "verifier transport failed")
                return
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = proxy
        do_POST = proxy

        def log_message(self, *_):
            pass  # Never print credentials or candidate payloads.

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
