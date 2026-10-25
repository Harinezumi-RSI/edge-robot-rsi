"""先生を OpenVLA-OFT にする: kitchen/oft_server.py(OFT の環境で起動)に HTTP で聞く。kitchen/vla.py の VLA(SmolVLA)と同じ使い方。

  act_batch(観測の列, 指示の列, active=...)  台所ごとに塊(既定 8 step)を使い切ってから聞き直す。終わった台所には聞かない
  chunk_batch(観測の列, 指示の列, n=8)        DAgger の答え: その場面から動き出すときの塊の先頭 n step → [N, n, 7]
"""
import base64
import json
import urllib.request

import numpy as np


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


class OFTTeacher:
    def __init__(self, url="http://127.0.0.1:8770", exec_k=8):
        self.url = url
        self.n_action_steps = exec_k
        self.num_steps = None
        self._q = {}
        self.last_sec = 0.0

    def _post(self, obs_list, instructions):
        items = [{"front": _b64(o["pixels"]["image"]), "wrist": _b64(o["pixels"]["image2"]), "shape": list(o["pixels"]["image"].shape),
                  "pos": np.asarray(o["robot_state"]["eef"]["pos"], float).tolist(),
                  "quat": np.asarray(o["robot_state"]["eef"]["quat"], float).tolist(),
                  "grip": np.asarray(o["robot_state"]["gripper"]["qpos"], float).tolist(), "instruction": s}
                 for o, s in zip(obs_list, instructions)]
        req = urllib.request.Request(self.url + "/chunk", data=json.dumps({"items": items}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read())
        self.last_sec = out["sec"]
        return np.asarray(out["actions"], dtype=np.float32)

    def reset(self):
        self._q = {}

    def act_batch(self, obs_list, instructions, active=None):
        active = [True] * len(obs_list) if active is None else list(active)
        need = [i for i, on in enumerate(active) if on and not self._q.get(i)]
        if need:
            ch = self._post([obs_list[i] for i in need], [instructions[i] for i in need])
            for i, c in zip(need, ch):
                self._q[i] = list(c[: self.n_action_steps])
        return np.stack([self._q[i].pop(0) if on else np.zeros(7, np.float32) for i, on in enumerate(active)])

    def chunk_batch(self, obs_list, instructions, n=8, seed=None):
        return self._post(obs_list, instructions)[:, :n]
