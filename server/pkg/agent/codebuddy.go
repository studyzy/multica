package agent

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// codebuddyBlockedArgs are flags hardcoded by the daemon that must not be
// overridden by user-configured custom_args. `--acp` is the protocol
// subcommand that drives the ACP JSON-RPC transport for CodeBuddy Code
// CLI; overriding it would break the daemon↔CodeBuddy communication
// contract.
var codebuddyBlockedArgs = map[string]blockedArgMode{
	"--acp": blockedStandalone,
}

// codebuddyBackend implements Backend by spawning `codebuddy --acp` and
// communicating via the ACP (Agent Client Protocol) JSON-RPC 2.0 over
// stdin/stdout.
//
// CodeBuddy Code (a Claude Code fork) supports ACP out of the box via the
// `codebuddy --acp` flag (see https://www.codebuddy.cn/docs/cli/acp). We
// reuse the existing hermesClient ACP transport since both runtimes speak
// the same protocol — only the binary, env, root-level CLI flags, and
// tool-name extraction differ.
type codebuddyBackend struct {
	cfg Config
}

// buildCodebuddyArgs assembles the argv for `codebuddy`. `--acp` switches
// the CLI into ACP server mode; the remaining root-level flags
// (`--effort`, `--max-turns`, `--append-system-prompt`) coexist with
// `--acp` and configure the session startup. Model switching, resume,
// and MCP config are NOT passed here — they flow through ACP
// `session/set_model`, `session/resume`, and `session/new.mcpServers`
// respectively, matching how kimiBackend / hermesBackend work.
func buildCodebuddyArgs(opts ExecOptions, logger *slog.Logger) []string {
	args := []string{"--acp"}
	if opts.ThinkingLevel != "" {
		args = append(args, "--effort", opts.ThinkingLevel)
	}
	if opts.MaxTurns > 0 {
		args = append(args, "--max-turns", fmt.Sprintf("%d", opts.MaxTurns))
	}
	if opts.SystemPrompt != "" {
		args = append(args, "--append-system-prompt", opts.SystemPrompt)
	}
	args = append(args, filterCustomArgs(opts.CustomArgs, codebuddyBlockedArgs, logger)...)
	return args
}

func (b *codebuddyBackend) Execute(ctx context.Context, prompt string, opts ExecOptions) (*Session, error) {
	execPath := b.cfg.ExecutablePath
	if execPath == "" {
		execPath = "codebuddy"
	}
	if _, err := exec.LookPath(execPath); err != nil {
		return nil, fmt.Errorf("codebuddy executable not found at %q: %w", execPath, err)
	}

	// Translate the agent's mcp_config (Claude-style object of objects)
	// into the array shape ACP `session/new` expects. Fail closed on
	// malformed JSON so the launch surfaces the real error instead of
	// silently dropping all MCP servers.
	mcpServers, err := buildACPMcpServers(opts.McpConfig, b.cfg.Logger)
	if err != nil {
		return nil, fmt.Errorf("codebuddy: invalid mcp_config: %w", err)
	}

	timeout := opts.Timeout
	runCtx, cancel := runContext(ctx, timeout)

	// `codebuddy --acp` does not honour --permission-mode / --yolo; the
	// daemon auto-approves in hermesClient.handleAgentRequest by replying
	// "approve_for_session" to every session/request_permission request.
	codebuddyArgs := buildCodebuddyArgs(opts, b.cfg.Logger)
	cmd := exec.CommandContext(runCtx, execPath, codebuddyArgs...)
	hideAgentWindow(cmd)
	b.cfg.Logger.Info("agent command", "exec", execPath, "args", codebuddyArgs)
	if opts.Cwd != "" {
		cmd.Dir = opts.Cwd
	}
	cmd.Env = buildEnv(b.cfg.Env)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("codebuddy stdout pipe: %w", err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("codebuddy stdin pipe: %w", err)
	}
	// Forward stderr to the daemon log *and* sniff provider-level
	// errors out of it so we can surface them in the task result.
	// CodeBuddy's session/prompt still reports stopReason=end_turn when
	// the underlying HTTP call to the LLM returns 4xx/5xx, so without
	// this the daemon reports a misleading "empty output" and the
	// actionable error (expired token, rate limit, upstream 5xx, …)
	// stays buried in the daemon log.
	//
	// StderrPipe + an explicit copier give us a join point
	// (`stderrDone`) that fires before the failure-promotion decision;
	// see the matching comment in hermes.go for why the io.MultiWriter
	// form races with stopReason=end_turn under load.
	providerErr := newACPProviderErrorSniffer("codebuddy")
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("codebuddy stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start codebuddy: %w", err)
	}

	stderrSink := io.MultiWriter(newLogWriter(b.cfg.Logger, "[codebuddy:stderr] "), providerErr)
	stderrDone := make(chan struct{})
	go func() {
		defer close(stderrDone)
		_, _ = io.Copy(stderrSink, stderr)
	}()

	b.cfg.Logger.Info("codebuddy acp started", "pid", cmd.Process.Pid, "cwd", opts.Cwd)

	msgCh := make(chan Message, 256)
	resCh := make(chan Result, 1)

	var outputMu sync.Mutex
	var output strings.Builder

	promptDone := make(chan hermesPromptResult, 1)

	// Reuse the hermesClient ACP transport — CodeBuddy speaks the same protocol.
	c := &hermesClient{
		cfg:          b.cfg,
		stdin:        stdin,
		pending:      make(map[int]*pendingRPC),
		pendingTools: make(map[string]*pendingToolCall),
		onMessage: func(msg Message) {
			// hermesClient.handleToolCallStart has already mapped the
			// raw ACP title via hermesToolNameFromTitle — which covers
			// lowercase hermes-style titles ("read:", "patch
			// (replace)", …) but not capitalised CodeBuddy-style ones
			// ("Read file: …", "Run command: …"). Re-normalise so the
			// UI sees consistent snake_case identifiers across all
			// backends. No-op when the name is already normal form.
			if msg.Type == MessageToolUse {
				msg.Tool = codebuddyToolNameFromTitle(msg.Tool)
			}
			if msg.Type == MessageText {
				outputMu.Lock()
				output.WriteString(msg.Content)
				outputMu.Unlock()
			}
			trySend(msgCh, msg)
		},
		onPromptDone: func(result hermesPromptResult) {
			select {
			case promptDone <- result:
			default:
			}
		},
	}

	// Start reading stdout in background.
	readerDone := make(chan struct{})
	go func() {
		defer close(readerDone)
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			c.handleLine(line)
		}
		c.closeAllPending(fmt.Errorf("codebuddy process exited"))
	}()

	// Drive the ACP session lifecycle in a goroutine.
	go func() {
		defer cancel()
		defer close(msgCh)
		defer close(resCh)
		defer func() {
			stdin.Close()
			_ = cmd.Wait()
		}()

		startTime := time.Now()
		finalStatus := "completed"
		var finalError string
		var sessionID string

		// 1. Initialize handshake.
		initResult, err := c.request(runCtx, "initialize", map[string]any{
			"protocolVersion": 1,
			"clientInfo": map[string]any{
				"name":    "multica-agent-sdk",
				"version": "0.2.0",
			},
			"clientCapabilities": map[string]any{},
		})
		if err != nil {
			finalStatus = "failed"
			finalError = fmt.Sprintf("codebuddy initialize failed: %v", err)
			resCh <- Result{Status: finalStatus, Error: finalError, DurationMs: time.Since(startTime).Milliseconds()}
			return
		}

		// Drop MCP entries whose remote transport the runtime didn't
		// advertise. See the matching comment in hermes.go for the why —
		// shipping an http/sse entry to a stdio-only runtime tanks the
		// whole session/new.
		mcpServers = filterACPMcpServersByCapability(mcpServers, extractACPMcpCapabilities(initResult), "codebuddy", b.cfg.Logger)

		// 2. Create or resume a session.
		cwd := opts.Cwd
		if cwd == "" {
			cwd = "."
		}

		if opts.ResumeSessionID != "" {
			// Per ACP Session Setup, session/resume accepts mcpServers
			// and the runtime re-connects them as part of the resume.
			result, err := c.request(runCtx, "session/resume", map[string]any{
				"cwd":        cwd,
				"sessionId":  opts.ResumeSessionID,
				"mcpServers": mcpServers,
			})
			if err != nil {
				finalStatus = "failed"
				finalError = fmt.Sprintf("codebuddy session/resume failed: %v", err)
				resCh <- Result{Status: finalStatus, Error: finalError, DurationMs: time.Since(startTime).Milliseconds()}
				return
			}
			var changed bool
			sessionID, changed = resolveResumedSessionID(opts.ResumeSessionID, result)
			if changed {
				b.cfg.Logger.Warn("agent returned a different session id on resume — original was likely lost; continuing with the new id",
					"backend", "codebuddy",
					"requested", opts.ResumeSessionID,
					"actual", sessionID,
				)
			}
		} else {
			result, err := c.request(runCtx, "session/new", map[string]any{
				"cwd":        cwd,
				"mcpServers": mcpServers,
			})
			if err != nil {
				finalStatus = "failed"
				finalError = fmt.Sprintf("codebuddy session/new failed: %v", err)
				resCh <- Result{Status: finalStatus, Error: finalError, DurationMs: time.Since(startTime).Milliseconds()}
				return
			}
			sessionID = extractACPSessionID(result)
			if sessionID == "" {
				finalStatus = "failed"
				finalError = "codebuddy session/new returned no session ID"
				resCh <- Result{Status: finalStatus, Error: finalError, DurationMs: time.Since(startTime).Milliseconds()}
				return
			}
		}

		c.sessionID = sessionID
		b.cfg.Logger.Info("codebuddy session created", "session_id", sessionID)

		// 3. If the caller picked a model (via agent.model from the UI
		// dropdown), ask codebuddy to switch the session to it before we
		// send any prompt. CodeBuddy's ACP server exposes
		// `session/set_model`; we pass the chosen modelId through
		// verbatim. This MUST fail the task on error: silently falling
		// back to codebuddy's default model would let the user believe
		// their pick was honoured while the task actually ran on
		// something else.
		if opts.Model != "" {
			if _, err := c.request(runCtx, "session/set_model", map[string]any{
				"sessionId": sessionID,
				"modelId":   opts.Model,
			}); err != nil {
				b.cfg.Logger.Warn("codebuddy set_session_model failed", "error", err, "requested_model", opts.Model)
				finalStatus = "failed"
				finalError = fmt.Sprintf("codebuddy could not switch to model %q: %v", opts.Model, err)
				if opts.ResumeSessionID != "" && isACPSessionNotFound(err) {
					// On a resumed session with a model override, the
					// dead session surfaces here instead of at
					// session/prompt. Same fix as the prompt path
					// below: clear the id so the daemon's
					// resume-failure fallback retries fresh.
					b.cfg.Logger.Warn("resumed session not found at set_model time; clearing session id so the daemon retries fresh",
						"backend", "codebuddy",
						"session_id", sessionID,
					)
					sessionID = ""
				}
				resCh <- Result{
					Status:     finalStatus,
					Error:      finalError,
					DurationMs: time.Since(startTime).Milliseconds(),
					SessionID:  sessionID,
				}
				return
			}
			b.cfg.Logger.Info("codebuddy session model set", "model", opts.Model)
		}

		// 4. Send the prompt and wait for PromptResponse. SystemPrompt,
		// if any, was already passed via --append-system-prompt at
		// process launch — do NOT splice it into the user turn.
		_, err = c.request(runCtx, "session/prompt", map[string]any{
			"sessionId": sessionID,
			"prompt": []map[string]any{
				{"type": "text", "text": prompt},
			},
		})
		if err != nil {
			if runCtx.Err() == context.DeadlineExceeded {
				finalStatus = "timeout"
				finalError = fmt.Sprintf("codebuddy timed out after %s", timeout)
			} else if runCtx.Err() == context.Canceled {
				finalStatus = "aborted"
				finalError = "execution cancelled"
			} else {
				finalStatus = "failed"
				finalError = fmt.Sprintf("codebuddy session/prompt failed: %v", err)
				if opts.ResumeSessionID != "" && isACPSessionNotFound(err) {
					// The runtime echoes the requested id back from
					// session/resume even when the session is gone, so
					// the stale id only fails here, at prompt time.
					// Empty SessionID lets the daemon's resume-failure
					// fallback retry fresh and store the replacement id.
					b.cfg.Logger.Warn("resumed session not found at prompt time; clearing session id so the daemon retries fresh",
						"backend", "codebuddy",
						"session_id", sessionID,
					)
					sessionID = ""
				}
			}
		} else {
			select {
			case pr := <-promptDone:
				if pr.stopReason == "cancelled" {
					finalStatus = "aborted"
					finalError = "codebuddy cancelled the prompt"
				}
				c.usageMu.Lock()
				c.usage.InputTokens += pr.usage.InputTokens
				c.usage.OutputTokens += pr.usage.OutputTokens
				c.usageMu.Unlock()
			default:
			}
		}

		duration := time.Since(startTime)
		b.cfg.Logger.Info("codebuddy finished", "pid", cmd.Process.Pid, "status", finalStatus, "duration", duration.Round(time.Millisecond).String())

		stdin.Close()
		cancel()

		<-readerDone
		// Ensure the stderr copier has drained before consulting the
		// provider-error sniffer; see hermes.go for the failure mode.
		<-stderrDone

		outputMu.Lock()
		finalOutput := output.String()
		outputMu.Unlock()

		// Promote completed→failed when stderr or the agent text stream
		// show a terminal upstream-LLM failure (HTTP 4xx / rate-limit /
		// expired token). See the helper docs for the full signal set;
		// the key safety property is that transient per-attempt
		// warnings followed by a successful retry stay "completed".
		finalStatus, finalError = promoteACPResultOnProviderError(finalStatus, finalError, finalOutput, providerErr)

		c.usageMu.Lock()
		u := c.usage
		c.usageMu.Unlock()

		var usageMap map[string]TokenUsage
		if u.InputTokens > 0 || u.OutputTokens > 0 || u.CacheReadTokens > 0 {
			model := opts.Model
			if model == "" {
				model = "unknown"
			}
			usageMap = map[string]TokenUsage{model: u}
		}

		resCh <- Result{
			Status:     finalStatus,
			Output:     finalOutput,
			Error:      finalError,
			DurationMs: duration.Milliseconds(),
			SessionID:  sessionID,
			Usage:      usageMap,
		}
	}()

	return &Session{Messages: msgCh, Result: resCh}, nil
}

// codebuddyToolNameFromTitle normalises tool names emitted by CodeBuddy's
// ACP server into the snake_case identifiers the Multica UI expects.
//
// CodeBuddy (a Claude Code fork) follows the ACP spec where `title` is a
// short human-readable label such as "Read file: /path/to/foo.go" or
// "Run command: ls". hermesToolNameFromTitle upstream handles hermes'
// lowercase convention ("read:", "patch (replace)") but not CodeBuddy's
// capitalised format — so we get called on the already-mapped name from
// hermes and fix up anything that slipped through. Empty input returns "".
func codebuddyToolNameFromTitle(title string) string {
	t := strings.TrimSpace(title)
	if t == "" {
		return ""
	}

	// Strip everything after the first colon — ACP titles often look like
	// "Tool Name: argument detail" and we want only the tool name.
	if idx := strings.Index(t, ":"); idx > 0 {
		t = strings.TrimSpace(t[:idx])
	}

	lower := strings.ToLower(t)
	switch lower {
	case "read", "read file":
		return "read_file"
	case "write", "write file":
		return "write_file"
	case "edit", "patch":
		return "edit_file"
	case "shell", "bash", "terminal", "run command", "run shell command":
		return "terminal"
	case "search", "grep", "find":
		return "search_files"
	case "glob":
		return "glob"
	case "web search":
		return "web_search"
	case "fetch", "web fetch":
		return "web_fetch"
	case "todo", "todo write":
		return "todo_write"
	}

	// Fallback: snake_case the title so the UI gets a stable identifier.
	return strings.ReplaceAll(lower, " ", "_")
}
