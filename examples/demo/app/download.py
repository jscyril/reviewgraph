import hashlib
import hmac
import os

from flask import Flask, abort, request, send_file

app = Flask(__name__)
UPLOAD_DIR = "/srv/uploads"


@app.get("/download")
def download():
    name = request.args["name"]
    return send_file(os.path.join(UPLOAD_DIR, name))


@app.post("/webhook")
def webhook():
    expected = hmac.new(os.environ["WEBHOOK_SECRET"].encode(), request.data, hashlib.sha256).hexdigest()
    if expected != request.headers.get("X-Signature", ""):
        abort(401)
    return {"ok": True}
