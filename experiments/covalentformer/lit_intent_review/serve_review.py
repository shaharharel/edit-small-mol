#!/usr/bin/env python3
"""Serve the intent-review app and PERSIST every verdict to this machine's disk.

The whole point of a server here rather than a static page: a verdict typed into a browser and
kept in localStorage is not a record. It lives in one person's browser profile, vanishes with a
cleared cache or a private window, and cannot be read back by anyone. So every submission is
appended to a JSONL file on the host that serves the page, with the reviewer, the timestamp and
the item id, and the browser copy is a convenience mirror only.

APPEND, NEVER REWRITE. A reviewer who changes their mind produces a second line for the same
item; the scorer takes the last line per (reviewer, item). Overwriting in place would make a
changed verdict indistinguishable from a first one and would lose the history of the change.

Usage:
  python serve_review.py --dir app --out reviews.jsonl --port 8765
  # on a VM, bind 0.0.0.0 and reach it over an SSH tunnel rather than opening a firewall port:
  #   gcloud compute ssh <vm> -- -L 8765:localhost:8765
"""
import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
ARGS = None
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]{2,}$')


# A per-process secret: tokens die when the server restarts, which is the right lifetime for a
# review round and means no secret is ever written to disk.
SECRET = secrets.token_bytes(32)


def sha(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def make_token(email):
    return hmac.new(SECRET, email.encode(), hashlib.sha256).hexdigest()[:32]


def check_token(email, token):
    return bool(token) and hmac.compare_digest(make_token(email), token)


def reviewers():
    """-> {email: sha256(password)}. One `email:sha256hex` per line, '#' comments ignored.

    PASSWORDS ARE STORED AS HASHES, not plaintext: the file sits on a VM several people can
    reach, and a reviewer list is not worth leaking a password someone reuses elsewhere.
    """
    path = getattr(ARGS, 'allow_file', None)
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if not line or ':' not in line:
            continue
        email, h = line.split(':', 1)
        out[email.strip().lower()] = h.strip()
    return out


def allowed_emails():
    """The reviewers permitted to submit, one per line, '#' comments ignored.

    WHY AN ALLOWLIST AND NOT A FREE-TEXT NAME. The page is on the open internet (port 5055 is
    0.0.0.0/0 so the LB can reach it), so anything that identifies a reviewer by a typed string
    can be typed by anyone, and a verdict whose author cannot be trusted is not a measurement.
    An empty or absent file means NO ONE is allowed -- fail closed, because the failure mode of
    failing open is a polluted record that looks exactly like a real one.
    """
    return set(reviewers())


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a):
        # EVERY REQUEST IS LOGGED, and the filter that used to live here is gone. It read
        # `'POST' in a[0]`, and send_error() calls this with an HTTPStatus as a[0], so a single
        # 404 (a browser asking for /favicon.ico) raised TypeError inside the handler and took
        # the request down with it. Two real verdicts were lost that way: a quiet log is not
        # worth a server that breaks on its own error path.
        try:
            super().log_message(fmt, *a)
        except Exception:
            pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip('/').endswith('whoami'):
            return self._json(200, {'n_allowed': len(allowed_emails())})
        if self.path.rstrip('/').endswith('favicon.ico'):
            self.send_response(204)
            self.end_headers()
            return
        if self.path.rstrip('/').endswith('where'):
            return self._json(200, {'file': os.path.abspath(ARGS.out),
                                    'host': socket.gethostname()})
        if self.path.rstrip('/').endswith('reviews'):
            # read-back endpoint, so progress can be checked without shell access to the box
            n, last = 0, None
            if os.path.exists(ARGS.out):
                for line in open(ARGS.out):
                    if line.strip():
                        n += 1
                        last = line
            return self._json(200, {'n': n, 'last': json.loads(last) if last else None})
        return super().do_GET()

    def do_POST(self):
        path = self.path.rstrip('/')
        if not (path.endswith('save') or path.endswith('auth')):
            return self._json(404, {'error': 'unknown endpoint'})
        try:
            n = int(self.headers.get('Content-Length') or 0)
            rec = json.loads(self.rfile.read(n) or b'{}')
        except Exception as e:
            return self._json(400, {'error': 'bad json: %s' % e})

        email = str(rec.get('email') or rec.get('reviewer') or '').strip().lower()
        people = reviewers()
        if not EMAIL_RE.match(email):
            return self._json(400, {'error': 'that does not look like an email address'})

        if path.endswith('auth'):
            # ONE message for a wrong address and a wrong password alike: telling a caller which
            # half they got right turns the reviewer list into something enumerable.
            pw = str(rec.get('password') or '')
            if email not in people or not hmac.compare_digest(people[email], sha(pw)):
                return self._json(403, {'error': 'email or password not recognised'})
            return self._json(200, {'ok': True, 'email': email, 'token': make_token(email)})

        if email not in people or not check_token(email, str(rec.get('token') or '')):
            # a token dies with the server, so this also catches a page left open across a
            # restart -- the reviewer signs in again rather than writing unattributable rows
            return self._json(401, {'error': 'session expired, sign in again'})
        if not rec.get('id') or rec.get('verdict') not in ('approve', 'reject', 'unsure'):
            return self._json(400, {'error': 'need id and verdict approve|reject|unsure'})
        rec['reviewer'] = email
        rec['received_at'] = datetime.now(timezone.utc).isoformat()
        rec['remote'] = self.client_address[0]
        with LOCK:
            with open(ARGS.out, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                os.fsync(fh.fileno())      # a verdict that is only in the page cache is not saved
            mine = set()
            for line in open(ARGS.out):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (row.get('reviewer') or '').lower() == email:
                    mine.add(row.get('id'))
        return self._json(200, {'ok': True, 'n': len(mine), 'file': os.path.abspath(ARGS.out)})


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='app')
    ap.add_argument('--out', default='reviews.jsonl')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--bind', default='127.0.0.1')
    ap.add_argument('--allow-file', dest='allow_file', default='reviewers.txt',
                    help='one reviewer email per line; absent or empty means nobody may submit')
    ARGS = ap.parse_args()
    ARGS.allow_file = os.path.abspath(os.path.expanduser(ARGS.allow_file))
    ARGS.dir = os.path.abspath(os.path.expanduser(ARGS.dir))
    ARGS.out = os.path.abspath(os.path.expanduser(ARGS.out))
    os.chdir(ARGS.dir)
    srv = ThreadingHTTPServer((ARGS.bind, ARGS.port), Handler)
    print('serving %s on http://%s:%d' % (ARGS.dir, ARGS.bind, ARGS.port))
    print('verdicts append to %s on %s' % (ARGS.out, socket.gethostname()))
    print('reviewer allowlist: %s (%d addresses)' % (ARGS.allow_file, len(allowed_emails())))
    srv.serve_forever()


if __name__ == '__main__':
    main()
