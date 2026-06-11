# Deploying the RLHF preference-collection server

The app is a Flask WSGI app served by **waitress**. SQLite is the datastore (fine
for a small labeling team). This doc covers running it as a service on a GCP VM and
the path to a real HTTPS domain.

## 1. One-line run (any host with the `quris` env)

From the repo root:

```bash
RLHF_SECRET_KEY=$(openssl rand -hex 32) conda run -n quris \
    python experiments/rlhf_server/serve.py
# -> http://0.0.0.0:5055
```

Environment variables (all optional except the secret in prod):

| var | default | purpose |
|---|---|---|
| `RLHF_SECRET_KEY` | dev key (insecure) | **set in prod** — signs session cookies |
| `RLHF_HOST` | `0.0.0.0` | bind host |
| `RLHF_PORT` | `5055` | bind port |
| `RLHF_THREADS` | `8` | worker threads |
| `RLHF_DB_PATH` | `data/rlhf_demo/rlhf.db` | SQLite path |

## 2. Data the server needs at runtime

The app only needs (NOT the big source CSV — that's build-time only):

- `data/rlhf_demo/molecules.json`, `data/rlhf_demo/pairs.json`
- `data/rlhf_demo/boltz_poses/<NN_CHEMBLID>/*_model_0.cif` (+ confidence json)

These are **gitignored**, so ship them out-of-band (scp/tar). To regenerate the
JSON on a host that has the source CSV + poses: `python build_pairs.py`.

## 3. Run as a systemd service (auto-start, auto-restart)

`/etc/systemd/system/rlhf.service`:

```ini
[Unit]
Description=RLHF preference-collection server
After=network.target

[Service]
User=shaharh_quris_ai
WorkingDirectory=/home/shaharh_quris_ai/edit-small-mol
Environment=RLHF_SECRET_KEY=<paste a 64-hex secret>
Environment=RLHF_PORT=5055
Environment=RLHF_THREADS=8
ExecStart=/home/shaharh_quris_ai/miniconda3/envs/quris/bin/python experiments/rlhf_server/serve.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rlhf
sudo systemctl status rlhf          # check
journalctl -u rlhf -f               # logs
```

## 4. Expose it (GCP firewall)

```bash
# open the app port to the chosen source range (lock down to office IPs if possible)
gcloud compute firewall-rules create allow-rlhf \
  --allow tcp:5055 --target-tags rlhf --source-ranges 0.0.0.0/0
gcloud compute instances add-tags ai-experiment --zone us-central1-b --tags rlhf
# -> http://<VM_EXTERNAL_IP>:5055
```

> ⚠️ **Auth caveat:** the app has no password — anyone who can reach it logs in with
> any email. On a public IP, restrict `--source-ranges` to trusted networks, or put
> it behind **Google IAP** before sharing widely (see §6).

## 5. Updating after a code change

```bash
cd ~/edit-small-mol && git pull
sudo systemctl restart rlhf
```
(Template/Python changes need the restart; the dev server caches templates too.)

## 6. Making it a real HTTPS website (domain + TLS)

1. **Subdomain**: point an `A` record `label.quris.ai → <VM static IP>` (reserve a
   static IP: `gcloud compute addresses create rlhf-ip --region us-central1`).
2. **nginx** reverse proxy in front of waitress (proxy_pass to 127.0.0.1:5055,
   serve `/static` directly).
3. **TLS**: `sudo certbot --nginx -d label.quris.ai` (Let's Encrypt, auto-renew).
4. Change waitress to bind `127.0.0.1` (only nginx talks to it) and close the
   public 5055 firewall rule.
5. **Real auth**: front the whole site with **Google IAP** (restrict to `@quris.ai`)
   — gives verified identity and removes the open-login caveat with no app code.

## 7. Backups

SQLite is one file. Back it up while live with the online backup API:

```bash
sqlite3 data/rlhf_demo/rlhf.db ".backup '/tmp/rlhf-$(date +%F).db'"
gsutil cp /tmp/rlhf-*.db gs://<your-bucket>/rlhf-backups/
```
(Or run `litestream` for continuous replication to GCS.)
