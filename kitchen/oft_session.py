"""OFT の先生を使うあいだだけ、Mac の共有の鍵を持ってサーバを立てる。
OFT は約 16GB。同じ Mac のほかの評価も同じ鍵の下で OFT を載せるので、24GB に 2 本載らないよう、鍵を持っている間だけサーバを生かす。

  python kitchen/oft_session.py -- <コマンド ...>

鍵を取る(PAI_MAC_LOCK、空なら取らない)→ OFT のサーバ(kitchen/oft_server.py)を立てる → コマンドを PAI_MAC_LOCK なしで走らせる
(中で鍵を取り直さない)→ サーバを止めて鍵を返す。終了コードはコマンドのもの。
OFT の環境の python と、prismatic・transformers の改造版の置き場は PAI_OFT_PY・PAI_OFT_CODE で変えられる。
"""
import contextlib
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFT_PY = os.environ.get("PAI_OFT_PY", os.path.join(ROOT, ".venv-oft", "bin", "python"))  # 用意の手順は README
OFT_CODE = os.environ.get("PAI_OFT_CODE", os.path.join(ROOT, "vendor", "openvla-oft"))  # prismatic の置き場(改造版 transformers は .venv-oft に入れる)
MODEL = os.path.join(ROOT, "models", "oft_goal")
PORT = int(os.environ.get("PAI_OFT_PORT", "8770"))


def say(*a):
    print(time.strftime("%m/%d %H:%M"), *a, flush=True)


@contextlib.contextmanager
def held_lock():
    lock = os.environ.get("PAI_MAC_LOCK", "")
    if not lock:
        yield
        return
    t = time.time()
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            time.sleep(30)
    me = f"paifinal {os.getpid()}"
    open(os.path.join(lock, "owner"), "w").write(me)
    say(f"[鍵] OFT を使う間持つ(待ち {(time.time() - t) / 60:.1f} 分)")
    try:
        yield
    finally:
        try:
            if open(os.path.join(lock, "owner")).read() == me:
                os.remove(os.path.join(lock, "owner"))
                os.rmdir(lock)
        except FileNotFoundError:
            pass
        say("[鍵] 返した")


@contextlib.contextmanager
def oft_server():
    log = open(os.path.join(ROOT, "logs", "oft_server.log"), "a")
    t = time.time()
    p = subprocess.Popen([OFT_PY, "kitchen/oft_server.py", "--model", MODEL, "--code", OFT_CODE, "--port", str(PORT)], cwd=ROOT,
                         env={"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1"},
                         stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            if p.poll() is not None:
                raise RuntimeError("OFT のサーバが落ちた(logs/oft_server.log)")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5).read()
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("OFT のサーバが 20 分で立たない")
        say(f"OFT のサーバが立った({time.time() - t:.0f} 秒)")
        yield f"http://127.0.0.1:{PORT}"
    finally:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
        say("OFT のサーバを止めた")


def main():
    cmd = sys.argv[sys.argv.index("--") + 1:]
    with held_lock(), oft_server() as url:
        rc = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, PAI_MAC_LOCK="", PAI_OFT_URL=url))
    sys.exit(rc)


if __name__ == "__main__":
    main()
