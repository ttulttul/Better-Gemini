import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ROUTE_PREFIX = "/better-gemini/grok/oauth";
const REQUEST_HEADER = { "X-Better-Gemini-Request": "1" };
const BUTTON_ATTRIBUTE = "data-better-grok-oauth";
const MODAL_ID = "better-grok-oauth-modal";
const STYLE_ID = "better-grok-oauth-styles";

let authState = { authenticated: false, login: null };
let statusLoaded = false;
let statusRequest = null;
let nodeObserver = null;
let syncQueued = false;
let pollingFlowId = null;
let initiatingNode = null;
let frontendInitialized = false;
const fallbackWidgets = new Set();

function toast(severity, summary, detail) {
  app.extensionManager?.toast?.add?.({ severity, summary, detail, life: 7000 });
}

async function responseJson(response) {
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    // The error below still reports the HTTP status without reflecting response text.
  }
  if (!response.ok) {
    throw new Error(payload.error || `BetterGrok OAuth request failed (${response.status}).`);
  }
  return payload;
}

async function fetchStatus() {
  if (statusRequest) return statusRequest;
  statusRequest = api
    .fetchApi(`${ROUTE_PREFIX}/status`)
    .then(responseJson)
    .then((state) => {
      authState = state;
      statusLoaded = true;
      renderButtons();
      if (state.login?.state === "pending") pollLogin(state.login.flow_id);
      return state;
    })
    .catch((error) => {
      console.warn("[BetterGrok OAuth] Unable to load login status:", error);
      return authState;
    })
    .finally(() => {
      statusRequest = null;
    });
  return statusRequest;
}

function nodeForContainer(container) {
  const rawId = container?.dataset?.nodeId;
  if (rawId == null) return null;
  return app.graph?.getNodeById?.(Number(rawId)) ?? app.graph?.getNodeById?.(rawId) ?? null;
}

function isBetterGrokNode(node) {
  return node?.comfyClass === "BetterGrok" || node?.type === "BetterGrok";
}

function buttonLabel() {
  if (!statusLoaded) return "Login";
  if (authState.authenticated) return "Logout";
  if (authState.login?.state === "pending") return "Waiting…";
  return "Login";
}

function renderButtons() {
  const label = buttonLabel();
  document.querySelectorAll(`[${BUTTON_ATTRIBUTE}]`).forEach((button) => {
    if (button.textContent !== label) button.textContent = label;
    button.dataset.state = authState.authenticated
      ? "authenticated"
      : authState.login?.state || "logged-out";
    button.disabled = authState.login?.state === "pending";
    button.title = authState.authenticated
      ? "Remove the saved xAI OAuth login"
      : "Sign in to xAI with device authorization";
  });
  fallbackWidgets.forEach((widget) => {
    widget.name = label;
  });
}

function installFallbackWidget(node) {
  if (!isBetterGrokNode(node) || node.__betterGrokOAuthWidget) return;
  const widget = node.addWidget?.("button", buttonLabel(), null, () => {
    handleAuthClick(node).catch((error) => {
      toast("error", "BetterGrok login", error.message);
      console.error("[BetterGrok OAuth] Login action failed:", error);
    });
  });
  if (!widget) return;
  node.__betterGrokOAuthWidget = widget;
  fallbackWidgets.add(widget);
  node.setSize?.([node.size[0], node.computeSize?.()[1] ?? node.size[1]]);
  app.graph?.setDirtyCanvas?.(true, true);
}

function scheduleRendererButton(node) {
  window.setTimeout(() => {
    const hasDomNode = [...document.querySelectorAll(".lg-node[data-node-id]")].some(
      (container) => nodeForContainer(container) === node,
    );
    if (hasDomNode) {
      queueButtonSync();
    } else {
      installFallbackWidget(node);
    }
  }, 250);
}

function installButton(container) {
  if (container.querySelector(`[${BUTTON_ATTRIBUTE}]`)) return;
  const node = nodeForContainer(container);
  if (!isBetterGrokNode(node)) return;
  const headerRow = container.querySelector(".lg-node-header > div");
  if (!headerRow) return;

  const button = document.createElement("button");
  button.type = "button";
  button.setAttribute(BUTTON_ATTRIBUTE, "");
  button.textContent = buttonLabel();
  for (const eventName of ["pointerdown", "mousedown", "dblclick"]) {
    button.addEventListener(eventName, (event) => event.stopPropagation());
  }
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    handleAuthClick(node).catch((error) => {
      toast("error", "BetterGrok login", error.message);
      console.error("[BetterGrok OAuth] Login action failed:", error);
    });
  });
  headerRow.append(button);
  renderButtons();
}

function syncButtons() {
  syncQueued = false;
  document.querySelectorAll(".lg-node[data-node-id]").forEach(installButton);
}

function queueButtonSync() {
  if (syncQueued) return;
  syncQueued = true;
  requestAnimationFrame(syncButtons);
}

function installObserver() {
  if (nodeObserver || !document.body) return;
  nodeObserver = new MutationObserver(queueButtonSync);
  nodeObserver.observe(document.body, { childList: true, subtree: true });
  queueButtonSync();
}

function installStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    [${BUTTON_ATTRIBUTE}] {
      border: 1px solid color-mix(in srgb, var(--color-muted-foreground, #94a3b8) 45%, transparent);
      border-radius: 0.375rem;
      background: color-mix(in srgb, var(--component-node-widget-background, #273244) 90%, transparent);
      color: var(--color-node-component-slot-text, #f8fafc);
      cursor: pointer;
      flex: 0 0 auto;
      font: inherit;
      line-height: 1;
      min-width: 3.5rem;
      padding: 0.28rem 0.45rem;
    }
    [${BUTTON_ATTRIBUTE}]:hover { filter: brightness(1.15); }
    [${BUTTON_ATTRIBUTE}][data-state="authenticated"] { border-color: #22c55e; }
    [${BUTTON_ATTRIBUTE}]:disabled { cursor: wait; opacity: 0.75; }
    #${MODAL_ID} {
      align-items: center;
      background: rgba(2, 6, 23, 0.72);
      display: flex;
      inset: 0;
      justify-content: center;
      position: fixed;
      z-index: 100000;
    }
    #${MODAL_ID} .better-grok-oauth-card {
      background: var(--comfy-menu-bg, #18181b);
      border: 1px solid var(--border-color, #3f3f46);
      border-radius: 0.75rem;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
      color: var(--input-text, #f4f4f5);
      max-width: 30rem;
      padding: 1.25rem;
      width: calc(100vw - 2rem);
    }
    #${MODAL_ID} h2 { font-size: 1.1rem; margin: 0 0 0.75rem; }
    #${MODAL_ID} p { line-height: 1.45; margin: 0.5rem 0; }
    #${MODAL_ID} code {
      background: rgba(148, 163, 184, 0.15);
      border-radius: 0.4rem;
      display: block;
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      margin: 0.8rem 0;
      padding: 0.8rem;
      text-align: center;
      user-select: all;
    }
    #${MODAL_ID} .better-grok-oauth-actions { display: flex; gap: 0.6rem; justify-content: flex-end; margin-top: 1rem; }
    #${MODAL_ID} button, #${MODAL_ID} a {
      background: #2563eb;
      border: 0;
      border-radius: 0.4rem;
      color: white;
      cursor: pointer;
      font: inherit;
      padding: 0.55rem 0.8rem;
      text-decoration: none;
    }
    #${MODAL_ID} button[data-secondary] { background: #52525b; }
  `;
  document.head.append(style);
}

async function initializeFrontend() {
  if (frontendInitialized) return;
  frontendInitialized = true;
  installStyles();
  installObserver();
  await fetchStatus();
}

function closeLoginModal() {
  document.getElementById(MODAL_ID)?.remove();
}

function showLoginModal(login) {
  closeLoginModal();
  const overlay = document.createElement("div");
  overlay.id = MODAL_ID;
  const card = document.createElement("div");
  card.className = "better-grok-oauth-card";
  const heading = document.createElement("h2");
  heading.textContent = "Sign in to xAI";
  const instructions = document.createElement("p");
  instructions.textContent = "Complete the xAI page in your browser. Enter this code if prompted:";
  const code = document.createElement("code");
  code.textContent = login.user_code;
  const waiting = document.createElement("p");
  waiting.textContent = "Waiting for approval… You can close this dialog; login will continue in ComfyUI.";
  const actions = document.createElement("div");
  actions.className = "better-grok-oauth-actions";
  const open = document.createElement("a");
  open.href = login.verification_uri;
  open.target = "_blank";
  open.rel = "noopener noreferrer";
  open.textContent = "Open xAI";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "Copy code";
  copy.addEventListener("click", () => navigator.clipboard?.writeText(login.user_code));
  const close = document.createElement("button");
  close.type = "button";
  close.dataset.secondary = "";
  close.textContent = "Close";
  close.addEventListener("click", closeLoginModal);
  actions.append(copy, open, close);
  card.append(heading, instructions, code, waiting, actions);
  overlay.append(card);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeLoginModal();
  });
  document.body.append(overlay);
}

function setNodeToOAuth(node) {
  const widget = node?.widgets?.find((candidate) => candidate.name === "auth_mode");
  if (!widget) return;
  widget.value = "oauth";
  widget.callback?.("oauth", app.canvas, node, [0, 0], null);
  app.graph?.setDirtyCanvas?.(true, true);
}

async function handleAuthClick(node) {
  await fetchStatus();
  if (authState.authenticated) {
    if (!window.confirm("Log BetterGrok out of xAI on this ComfyUI server?")) return;
    const response = await api.fetchApi(`${ROUTE_PREFIX}/logout`, {
      method: "POST",
      headers: REQUEST_HEADER,
    });
    authState = await responseJson(response);
    pollingFlowId = null;
    closeLoginModal();
    renderButtons();
    toast("success", "BetterGrok", "Logged out of xAI.");
    return;
  }

  initiatingNode = node;
  const pendingWindow = window.open("", "_blank");
  let response;
  try {
    response = await api.fetchApi(`${ROUTE_PREFIX}/login`, {
      method: "POST",
      headers: REQUEST_HEADER,
    });
  } catch (error) {
    pendingWindow?.close();
    throw error;
  }
  authState = await responseJson(response);
  renderButtons();
  const login = authState.login;
  if (pendingWindow) {
    pendingWindow.opener = null;
    pendingWindow.location.replace(login.verification_uri);
  }
  showLoginModal(login);
  pollLogin(login.flow_id);
}

async function pollLogin(flowId) {
  if (!flowId || pollingFlowId === flowId) return;
  pollingFlowId = flowId;
  try {
    while (pollingFlowId === flowId) {
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
      const response = await api.fetchApi(`${ROUTE_PREFIX}/login/${encodeURIComponent(flowId)}`);
      const state = await responseJson(response);
      authState = state;
      renderButtons();
      if (state.login?.state === "pending") continue;
      if (state.authenticated || state.login?.state === "authenticated") {
        setNodeToOAuth(initiatingNode);
        closeLoginModal();
        toast("success", "BetterGrok", "Logged in to xAI. This node now uses OAuth.");
      } else {
        closeLoginModal();
        toast("error", "BetterGrok login", state.login?.error || "xAI login failed.");
      }
      break;
    }
  } catch (error) {
    toast("error", "BetterGrok login", error.message);
  } finally {
    if (pollingFlowId === flowId) pollingFlowId = null;
  }
}

app.registerExtension({
  name: "BetterGemini.GrokOAuth",

  async init() {
    await initializeFrontend();
  },

  async setup() {
    await initializeFrontend();
  },

  async nodeCreated(node) {
    if (isBetterGrokNode(node)) scheduleRendererButton(node);
  },

  async afterConfigureGraph() {
    queueButtonSync();
    for (const node of app.graph?._nodes ?? []) {
      if (isBetterGrokNode(node)) scheduleRendererButton(node);
    }
  },
});
