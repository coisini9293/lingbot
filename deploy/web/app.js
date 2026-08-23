/* pot14 console — GSAP entrances + live cameras + rate limits */
(() => {
  const els = {
    modeBtns: [...document.querySelectorAll(".mode-btn")],
    serialPanel: document.getElementById("serialPanel"),
    wsUrl: document.getElementById("wsUrl"),
    task: document.getElementById("task"),
    steps: document.getElementById("steps"),
    actionFps: document.getElementById("actionFps"),
    maxJointStep: document.getElementById("maxJointStep"),
    executeChunk: document.getElementById("executeChunk"),
    serialPort: document.getElementById("serialPort"),
    camTop: document.getElementById("camTop"),
    camLeft: document.getElementById("camLeft"),
    camRight: document.getElementById("camRight"),
    liveTop: document.getElementById("liveTop"),
    liveLeft: document.getElementById("liveLeft"),
    liveRight: document.getElementById("liveRight"),
    deviceBox: document.getElementById("deviceBox"),
    logBox: document.getElementById("logBox"),
    summaryBox: document.getElementById("summaryBox"),
    btnRun: document.getElementById("btnRun"),
    btnPorts: document.getElementById("btnPorts"),
    btnCams: document.getElementById("btnCams"),
    btnStopPreview: document.getElementById("btnStopPreview"),
    btnPrep: document.getElementById("btnPrep"),
    btnJog: document.getElementById("btnJog"),
    mockState: document.getElementById("mockState"),
    previewHint: document.getElementById("previewHint"),
    linkStatus: document.getElementById("linkStatus"),
    linkStatusText: document.getElementById("linkStatusText"),
    runBadge: document.getElementById("runBadge"),
  };

  let mode = "test";
  let cameras = [];
  let streaming = false;
  let autoPreview = true; // 准备实机后关闭，避免抢相机

  function setMode(next) {
    mode = next;
    els.modeBtns.forEach((btn) => {
      const active = btn.dataset.mode === next;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    const showSerial = next === "serial";
    if (showSerial) {
      els.serialPanel.hidden = false;
      if (window.gsap) {
        gsap.matchMedia().add("(prefers-reduced-motion: no-preference)", () => {
          gsap.fromTo(
            els.serialPanel,
            { autoAlpha: 0, y: 16 },
            { autoAlpha: 1, y: 0, duration: 0.45, ease: "power3.out" }
          );
        });
      }
      refreshPorts();
      refreshCameras().then(() => {
        if (autoPreview) startLiveStreams();
      });
    } else {
      stopLiveStreams();
      els.serialPanel.hidden = true;
    }
  }

  els.modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  function setStatus(kind, text) {
    els.linkStatus.classList.remove("is-live", "is-run");
    if (kind === "live") els.linkStatus.classList.add("is-live");
    if (kind === "run") els.linkStatus.classList.add("is-run");
    els.linkStatusText.textContent = text;
  }

  function setBadge(kind, text) {
    els.runBadge.className = "badge" + (kind ? ` is-${kind}` : "");
    els.runBadge.textContent = text;
  }

  async function getJSON(url, options) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText || "request failed");
    return data;
  }

  function selectedIndices() {
    return [
      Number(els.camTop.value),
      Number(els.camLeft.value),
      Number(els.camRight.value),
    ];
  }

  function fillPortSelect(ports) {
    const prev = els.serialPort.value;
    els.serialPort.innerHTML = "";
    if (!ports.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "未找到串口 — 点「刷新串口」重试";
      els.serialPort.appendChild(opt);
      return;
    }
    ports.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      const short = p.split("/").pop() || p;
      opt.textContent = short.startsWith("cu.") ? `${p}  ← 推荐` : p;
      els.serialPort.appendChild(opt);
    });
    if (prev && ports.includes(prev)) {
      els.serialPort.value = prev;
    } else {
      // 默认选第一个（已按 cu.* 优先排序）
      els.serialPort.value = ports[0];
    }
  }

  async function refreshPorts() {
    try {
      const data = await getJSON("/api/ports");
      const ports = data.ports || [];
      fillPortSelect(ports);
      const camNote = els.deviceBox.textContent || "";
      els.deviceBox.textContent =
        (ports.length
          ? "串口：\n" + ports.map((p) => `• ${p}`).join("\n")
          : "串口：未找到") +
        (camNote.includes("index=") ? "\n\n" + camNote : "");
      pulse(els.deviceBox);
      return ports;
    } catch (err) {
      els.deviceBox.textContent = String(err);
      fillPortSelect([]);
      throw err;
    }
  }

  function fillCamSelects(suggested) {
    const selects = [els.camTop, els.camLeft, els.camRight];
    selects.forEach((sel, i) => {
      const prev = sel.value;
      sel.innerHTML = "";
      if (!cameras.length) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `未找到摄像头`;
        sel.appendChild(opt);
        return;
      }
      cameras.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = String(c.index);
        const uid = (c.unique_id || "").slice(-8);
        opt.textContent = `avf ${c.index} · ${c.name || "cam"} · …${uid}`;
        sel.appendChild(opt);
      });
      // 默认：三个下拉分别选不同的前三路，避免都选同一路
      let prefer = prev;
      if (!(prefer !== "" && [...sel.options].some((o) => o.value === prefer))) {
        prefer = String(
          suggested[i] ?? cameras[Math.min(i, cameras.length - 1)].index
        );
      }
      sel.value = prefer;
    });
    // 若默认撞车（同一 avf），尽量拆开
    const used = new Set();
    selects.forEach((sel) => {
      if (!used.has(sel.value)) {
        used.add(sel.value);
        return;
      }
      const free = cameras.find((c) => !used.has(String(c.index)));
      if (free) {
        sel.value = String(free.index);
        used.add(sel.value);
      }
    });
  }

  function stopLiveStreams() {
    streaming = false;
    [els.liveTop, els.liveLeft, els.liveRight].forEach((img) => {
      img.removeAttribute("src");
      img.classList.remove("is-live");
    });
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function releasePreviewCams() {
    stopLiveStreams();
    try {
      await getJSON("/api/cameras/release", { method: "POST" });
    } catch {
      /* 旧服务端无此接口时忽略 */
    }
    await sleep(200);
  }

  function startLiveStreams() {
    if (mode !== "serial") return;
    const idxs = selectedIndices();
    const imgs = [els.liveTop, els.liveLeft, els.liveRight];
    streaming = true;
    imgs.forEach((img, i) => {
      const idx = idxs[i];
      if (Number.isNaN(idx)) return;
      img.src = `/api/camera/${idx}/mjpeg?t=${Date.now()}`;
      img.classList.add("is-live");
    });
  }

  async function refreshCameras() {
    try {
      // 预览流占着 USB 时，Mac 上再枚举会失败，只剩 FaceTime
      await releasePreviewCams();
      const data = await getJSON("/api/cameras");
      cameras = data.cameras || [];
      const suggested = data.suggested_indices || cameras.map((c) => c.index).slice(0, 3);
      fillCamSelects(suggested);
      const note = data.note ? `\n\n注意：${data.note}` : "";
      const camText = !cameras.length
        ? "摄像头：未找到（请退出 OBS，安装 pyobjc-framework-AVFoundation）" + note
        : "摄像头（AVFoundation，可在上方下拉为 top/left/right 任选）：\n" +
          cameras
            .map(
              (c) =>
                `• avf=${c.index}  ${c.name || "(无名)"}  uid=${c.unique_id || "-"}`
            )
            .join("\n") +
          note;
      const ports = [...els.serialPort.options]
        .map((o) => o.value)
        .filter(Boolean);
      els.deviceBox.textContent =
        (ports.length
          ? "串口：\n" + ports.map((p) => `• ${p}`).join("\n") + "\n\n"
          : "") + camText;
      pulse(els.deviceBox);
      return data;
    } catch (err) {
      els.deviceBox.textContent = String(err);
      throw err;
    }
  }

  [els.camTop, els.camLeft, els.camRight].forEach((sel) => {
    sel.addEventListener("change", () => {
      if (streaming) startLiveStreams();
    });
  });

  els.btnPorts.addEventListener("click", async () => {
    try {
      await refreshPorts();
    } catch {
      /* deviceBox 已展示错误 */
    }
  });

  els.btnCams.addEventListener("click", async () => {
    els.btnCams.disabled = true;
    autoPreview = true;
    try {
      await refreshCameras();
      startLiveStreams();
      if (els.previewHint) {
        els.previewHint.textContent =
          "预览已打开。正式跑之前请点「关闭预览」或「准备实机」。";
      }
    } finally {
      els.btnCams.disabled = false;
    }
  });

  async function closePreviewForRun() {
    autoPreview = false;
    await releasePreviewCams();
    setStatus("live", "预览已关，可开始");
    if (els.previewHint) {
      els.previewHint.textContent =
        "相机已释放。确认串口与任务文本后，点「开始运行」。";
    }
  }

  if (els.btnStopPreview) {
    els.btnStopPreview.addEventListener("click", async () => {
      els.btnStopPreview.disabled = true;
      try {
        await closePreviewForRun();
        els.deviceBox.textContent =
          (els.deviceBox.textContent || "") + "\n\n已关闭预览并释放相机。";
      } finally {
        els.btnStopPreview.disabled = false;
      }
    });
  }

  if (els.btnPrep) {
    els.btnPrep.addEventListener("click", async () => {
      els.btnPrep.disabled = true;
      try {
        await closePreviewForRun();
        const idxs = selectedIndices();
        const port = els.serialPort.value.trim();
        const lines = [
          "=== 实机准备检查 ===",
          `串口: ${port || "（未选）"}`,
          `相机 avf: top=${idxs[0]} left=${idxs[1]} right=${idxs[2]}`,
          `任务: ${els.task.value.trim()}`,
          `步数=${els.steps.value}  频率=${els.actionFps.value}Hz  步进=${els.maxJointStep.value}`,
          `零状态联调: ${els.mockState && els.mockState.checked ? "是" : "否"}`,
          "",
          "请确认：1) AutoDL deploy 在跑  2) OBS 已退出  3) 预览已关闭",
          "然后点「开始运行」。首次推理可能要 20～30 秒，请耐心等日志。",
        ];
        els.deviceBox.textContent = lines.join("\n");
        els.logBox.textContent = "已准备就绪，等待开始…\n";
        setBadge("", "ready");
      } finally {
        els.btnPrep.disabled = false;
      }
    });
  }

  if (els.btnJog) {
    els.btnJog.addEventListener("click", async () => {
      const port = els.serialPort.value.trim();
      if (!port) {
        els.logBox.textContent = "请先选择串口，再点硬件点动自检。\n";
        return;
      }
      els.btnJog.disabled = true;
      setStatus("busy", "点动自检中");
      els.logBox.textContent =
        "正在对右臂 P7 做缓升/缓降点动（不走模型）…\n请盯着机械臂看有没有动。\n";
      try {
        await closePreviewForRun();
        const data = await getJSON("/api/jog", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            serial_port: port,
            joint_index: 0,
            peak_delta_adc: 160,
            step_adc: 40,
            hold_seconds: 1.0,
          }),
        });
        els.logBox.textContent += JSON.stringify(data, null, 2) + "\n";
        if (data.ok) {
          setStatus("live", "点动结束，看手臂");
          setBadge("看手臂有无动作", "ok");
        } else {
          setStatus("err", "点动失败");
          setBadge(data.error || "fail", "err");
        }
      } catch (err) {
        els.logBox.textContent += String(err) + "\n";
        setStatus("err", "点动失败");
      } finally {
        els.btnJog.disabled = false;
      }
    });
  }

  els.btnRun.addEventListener("click", async () => {
    const idxs = selectedIndices();
    const payload = {
      mode,
      ws_url: els.wsUrl.value.trim(),
      task: els.task.value.trim(),
      steps: Number(els.steps.value) || 5,
      serial_port: els.serialPort.value.trim(),
      camera_indices: idxs,
      execute_chunk: !!els.executeChunk.checked,
      action_fps: Number(els.actionFps.value) || 15,
      max_joint_step: (() => {
        const v = Number(els.maxJointStep.value);
        return Number.isFinite(v) ? v : 0;
      })(),
      mock_state: !!(els.mockState && els.mockState.checked),
    };

    els.btnRun.disabled = true;
    setStatus("run", "运行中…首次推理较慢");
    setBadge("run", "running");
    els.logBox.textContent =
      "正在关闭预览并连接 AutoDL…\n首次推理可能 20～30 秒，请不要刷新页面。\n";
    autoPreview = false;
    await releasePreviewCams();
    shineRun();

    try {
      const data = await getJSON("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      els.logBox.textContent = data.log || "";
      els.summaryBox.textContent = JSON.stringify(data.summary || {}, null, 2);
      if (data.ok) {
        setStatus("live", "完成");
        setBadge("ok", "ok");
      } else {
        setStatus("live", "失败");
        setBadge("fail", "fail");
      }
      pulse(els.logBox);
    } catch (err) {
      els.logBox.textContent = String(err);
      setStatus("live", "失败");
      setBadge("fail", "fail");
    } finally {
      els.btnRun.disabled = false;
      // 不自动重开预览，避免抢相机；需要时点「刷新并打开预览」
      if (els.previewHint) {
        els.previewHint.textContent =
          "运行结束。若要再预览，点「刷新并打开预览」。";
      }
    }
  });

  function pulse(el) {
    if (!window.gsap) return;
    gsap.matchMedia().add("(prefers-reduced-motion: no-preference)", () => {
      gsap.fromTo(
        el,
        { outlineColor: "rgba(31,107,74,0.6)" },
        {
          outlineColor: "rgba(31,107,74,0)",
          duration: 0.8,
          ease: "power2.out",
        }
      );
    });
  }

  function shineRun() {
    if (!window.gsap) return;
    const shine = els.btnRun.querySelector(".run-shine");
    gsap.matchMedia().add("(prefers-reduced-motion: no-preference)", () => {
      gsap.fromTo(
        shine,
        { x: "-120%" },
        { x: "120%", duration: 0.9, ease: "power2.inOut" }
      );
    });
  }

  function intro() {
    if (!window.gsap) {
      document.querySelectorAll("[data-anim]").forEach((el) => {
        el.style.opacity = "1";
      });
      return;
    }
    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: reduce)", () => {
      gsap.set("[data-anim]", { autoAlpha: 1, y: 0 });
    });
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
      gsap.set("[data-anim]", { autoAlpha: 0, y: 28 });
      tl.to('[data-anim="hero"]', { autoAlpha: 1, y: 0, duration: 0.7 })
        .to(
          '[data-anim="panel"]',
          { autoAlpha: 1, y: 0, duration: 0.55, stagger: 0.08 },
          "-=0.35"
        )
        .to('[data-anim="foot"]', { autoAlpha: 1, y: 0, duration: 0.4 }, "-=0.2");
    });
  }

  async function loadDefaults() {
    try {
      const data = await getJSON("/api/defaults");
      if (data.ws_url) els.wsUrl.value = data.ws_url;
      if (data.task) els.task.value = data.task;
      if (data.steps != null) els.steps.value = data.steps;
      if (data.action_fps != null) els.actionFps.value = data.action_fps;
      if (data.max_joint_step != null) els.maxJointStep.value = data.max_joint_step;
      cameras = data.cameras || [];
      fillCamSelects(data.camera_indices || suggestFallback(cameras));
    } catch {
      /* keep HTML placeholders */
    }
  }

  function suggestFallback(cams) {
    const idxs = cams.map((c) => c.index).slice(0, 3);
    while (idxs.length < 3) idxs.push(idxs.length);
    return idxs;
  }

  loadDefaults().then(() => {
    intro();
    setMode("test");
  });
})();
