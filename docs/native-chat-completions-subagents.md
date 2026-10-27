# Native Chat Completions subagents

This implementation is based on upstream Codex commit
`75667b341af201f6e20a228437ebfe82b828b53c` (2026-10-28 checkout).

Codex keeps the main session on its selected provider and authentication path. A spawned agent role can select a separate `model` and `model_provider`; a provider with `wire_api = "chat"` is called directly at `/chat/completions` using that provider's configured API key. Chat providers are stateless, so Codex resends the mapped conversation history on each turn.

Add provider definitions and roles to `~/.codex/config.toml`:

```toml
[model_providers.deepseek]
name = "DeepSeek"
base_url = "https://api.deepseek.com"
env_key = "DEEPSEEK_API_KEY"
wire_api = "chat"

[model_providers.glm]
name = "GLM"
base_url = "https://open.bigmodel.cn/api/paas/v4"
env_key = "GLM_API_KEY"
wire_api = "chat"

[agents.implementer]
description = "Implement coding tasks"
model = "deepseek-flash"
model_provider = "deepseek"

[agents.reviewer]
description = "Review code using GLM"
model = "YOUR_GLM_CHAT_MODEL_ID"
model_provider = "glm"
```

Set `DEEPSEEK_API_KEY` and `GLM_API_KEY` in the environment that launches Codex. Replace `YOUR_GLM_CHAT_MODEL_ID` with a model ID available to your account. DeepSeek's current API docs use `deepseek-flash` and `deepseek-v4-pro`; older `deepseek-chat` references may no longer be valid. See [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) and [GLM API examples](https://docs.bigmodel.cn/cn/best-practice/case/ai-search-engine) for provider-specific endpoint and model details.

Role-level provider selection applies only to the child configuration. The main agent's provider and ChatGPT login remain unchanged. The Chat adapter supports text messages, inline images, function tools (including flattened namespaced functions), streamed text and fragmented tool calls, tool results, and usage when the provider sends it. Codex currently sends streamed requests only; non-streaming Chat completions are not implemented. It reports an explicit request error for Responses-only web search, structured output schemas, file-backed images, audio, freeform tools, and other Codex history items without a Chat equivalent. Provider-specific options outside the shared Chat request shape are not sent.

## Build and checks

Build the standalone CLI in this checkout with `cargo build --locked -p codex-cli` from `codex-rs/`; the binary is `target/debug/codex`. For ARM64 package generation, use the repository's existing CI/release workflow rather than overwriting the installed Codex app. Configuration lives in `~/.codex/config.toml`; set the provider's environment key before launching the built CLI.

Use the repository's existing Rust workflows: `.github/workflows/rust-ci.yml` for fast checks and `.github/workflows/rust-ci-full.yml` for the Linux x64, Linux ARM64, and Windows x64 platform matrix. The release workflows already include Linux ARM64 and Windows x64 targets. This checkout does not add a separate release workflow; GitHub Actions can only run after these changes are pushed to an authorized repository or fork.

Focused local verification on a 2 GiB Raspberry Pi:

```sh
cargo fmt --all -- --check
cargo test -p codex-model-provider-info
cargo test -p codex-api --test chat_completions
cargo test -p codex-core agent::role_tests::apply_role_selects_its_own_chat_provider_and_model
```

The full workspace test matrix should run on GitHub Actions. Check reports and downloadable artifacts from the workflow run before treating an architecture as verified.

## Windows GUI integration

The public Codex repository contains the app-server protocol and Windows app-server build targets, but does not expose a supported setting for choosing an arbitrary app-server executable in the installed Windows Codex GUI. OpenAI describes desktop clients as bundling or fetching a pinned platform Codex binary and launching it as a child process over stdio ([App Server overview](https://openai.com/index/unlocking-the-codex-harness/)). Keep this build as a standalone CLI/app-server or integrate it through a separately maintained client that speaks the [public app-server protocol](https://github.com/openai/codex/tree/main/codex-rs/app-server-protocol). Do not replace the installed app or patch its binaries.
