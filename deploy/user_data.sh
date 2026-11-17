#!/bin/bash
# EC2 first-boot script (cloud-init user data) for Ubuntu 24.04.
# Pulls the app bundle + .env from the private S3 bucket using the instance IAM role (no keys anywhere),
# installs dependencies, and runs Streamlit on 127.0.0.1 behind nginx with HTTPS.
# Progress log: /var/log/solstice-deploy.log   Finished marker: /opt/solstice/DEPLOY_OK
set -eux
exec > >(tee -a /var/log/solstice-deploy.log) 2>&1

BUCKET=YOUR_BUCKET_NAME   # replace with your private bucket (see docs/AWS_SETUP.md)
REGION=ap-south-1
APP=/opt/solstice
export DEBIAN_FRONTEND=noninteractive HOME=/root

# 1) swap: the instance is small and torch needs headroom while loading
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
fi

# 2) OS packages
apt-get update -y
apt-get install -y python3-venv python3-pip tesseract-ocr unzip nginx openssl

# 3) app bundle + .env from the PRIVATE bucket via short-lived presigned links (no credentials on the box).
#    __BUNDLE_URL__ / __ENV_URL__ are filled in by the launch step; the links expire after one hour.
#    At runtime the app reaches S3 only through the instance IAM role (attach it in the console).
mkdir -p $APP && cd $APP
python3 -m venv .venv
curl -fsSL "__BUNDLE_URL__" -o /tmp/solstice-app.zip
curl -fsSL "__ENV_URL__" -o $APP/.env
unzip -o /tmp/solstice-app.zip -d $APP

# 4) Python dependencies (CPU-only torch: the default Linux wheel pulls ~2 GB of CUDA libraries we cannot use)
.venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --no-cache-dir -r requirements.txt
export HF_HOME=$APP/.cache/huggingface
.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
chown -R ubuntu:ubuntu $APP
chmod 600 $APP/.env

# 5) Streamlit as a systemd service, bound to localhost only
cat > /etc/systemd/system/solstice.service <<UNIT
[Unit]
Description=Solstice Docs Assistant (Streamlit)
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=$APP
Environment=HF_HOME=$APP/.cache/huggingface
ExecStart=$APP/.venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

# 6) nginx: HTTPS with a self-signed certificate in front of Streamlit (browser will show a warning)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -subj "/CN=solstice-docs" \
  -keyout /etc/ssl/private/solstice.key -out /etc/ssl/certs/solstice.crt
rm -f /etc/nginx/sites-enabled/default
cat > /etc/nginx/conf.d/solstice.conf <<'NGINX'
server {
    listen 80 default_server;
    return 301 https://$host$request_uri;
}
server {
    listen 443 ssl default_server;
    ssl_certificate     /etc/ssl/certs/solstice.crt;
    ssl_certificate_key /etc/ssl/private/solstice.key;
    client_max_body_size 25m;
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
NGINX
nginx -t

systemctl daemon-reload
systemctl enable --now solstice
systemctl restart nginx
touch $APP/DEPLOY_OK
echo "DEPLOY COMPLETE"
