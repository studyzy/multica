package agent

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestBuildCodebuddyArgs_AcpRoot(t *testing.T) {
	t.Parallel()

	args := buildCodebuddyArgs(ExecOptions{}, slog.Default())

	// --acp is always present and first.
	if len(args) == 0 || args[0] != "--acp" {
		t.Fatalf("expected first arg --acp, got %v", args)
	}
	// No stream-json / -p / --permission-mode leakage.
	joined := strings.Join(args, " ")
	for _, banned := range []string{"-p ", "--output-format", "--input-format", "--permission-mode", "--mcp-config", "--disallowedTools", "--verbose", "--strict-mcp-config"} {
		if strings.Contains(joined, banned) {
			t.Fatalf("stream-json flag %q leaked into ACP args: %v", banned, args)
		}
	}
}

func TestBuildCodebuddyArgs_EffortMaxTurnsSystemPrompt(t *testing.T) {
	t.Parallel()

	args := buildCodebuddyArgs(ExecOptions{
		ThinkingLevel: "high",
		MaxTurns:      25,
		SystemPrompt:  "You are an agent.",
	}, slog.Default())

	want := []string{"--acp", "--effort", "high", "--max-turns", "25", "--append-system-prompt", "You are an agent."}
	if len(args) != len(want) {
		t.Fatalf("expected %d args, got %d: %v", len(want), len(args), args)
	}
	for i, w := range want {
		if args[i] != w {
			t.Fatalf("args[%d] = %q, want %q (full: %v)", i, args[i], w, args)
		}
	}
}

func TestBuildCodebuddyArgs_BlocksAcpOverride(t *testing.T) {
	t.Parallel()

	args := buildCodebuddyArgs(ExecOptions{
		CustomArgs: []string{"--acp", "--evil-flag"},
	}, slog.Default())

	count := 0
	for _, a := range args {
		if a == "--acp" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected exactly 1 --acp (daemon-injected), got %d in: %v", count, args)
	}
	// --evil-flag survives filtering (only --acp is blocked).
	found := false
	for _, a := range args {
		if a == "--evil-flag" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected --evil-flag to survive filtering: %v", args)
	}
}

func TestCodebuddyToolNameFromTitle(t *testing.T) {
	t.Parallel()
	tests := []struct {
		title string
		want  string
	}{
		{"Read file: /tmp/foo.go", "read_file"},
		{"read", "read_file"},
		{"Write: /tmp/bar.go", "write_file"},
		{"Edit", "edit_file"},
		{"Patch: /tmp/x", "edit_file"},
		{"Shell: ls -la", "terminal"},
		{"Bash", "terminal"},
		{"Run command: pwd", "terminal"},
		{"Search: foo", "search_files"},
		{"Glob: *.go", "glob"},
		{"Web search: golang acp", "web_search"},
		{"Fetch: https://example.com", "web_fetch"},
		{"Todo Write", "todo_write"},
		// Already-normalised input passes through unchanged.
		{"read_file", "read_file"},
		// Fallback: snake_case the title.
		{"Custom Thing", "custom_thing"},
		// Empty input returns empty — caller decides how to react.
		{"", ""},
	}
	for _, tt := range tests {
		got := codebuddyToolNameFromTitle(tt.title)
		if got != tt.want {
			t.Errorf("codebuddyToolNameFromTitle(%q) = %q, want %q", tt.title, got, tt.want)
		}
	}
}

// fakeCodebuddyACPScript returns a POSIX-sh script that impersonates
// `codebuddy --acp` for a single short ACP session: it acks initialize /
// session/new, optional session/set_model, and session/prompt with a
// text agent_message_chunk notification followed by a PromptResponse.
// Exits after prompt so the codebuddyBackend cleanup path can run.
func fakeCodebuddyACPScript() string {
	return `#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":1,"agentCapabilities":{}}}\n' "$id"
      ;;
    *'"method":"session/new"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"sessionId":"ses_cb_001"}}\n' "$id"
      ;;
    *'"method":"session/set_model"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      ;;
    *'"method":"session/prompt"'*)
      printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"ses_cb_001","update":{"type":"agent_message_chunk","content":{"type":"text","text":"Hello from codebuddy"}}}}\n'
      printf '{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"end_turn","usage":{"inputTokens":100,"outputTokens":50}}}\n' "$id"
      exit 0
      ;;
  esac
done
`
}

func TestCodebuddyExecute_Success(t *testing.T) {
	t.Parallel()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fixture is POSIX-only")
	}

	fakePath := filepath.Join(t.TempDir(), "codebuddy")
	writeTestExecutable(t, fakePath, []byte(fakeCodebuddyACPScript()))

	b := &codebuddyBackend{cfg: Config{ExecutablePath: fakePath, Logger: slog.Default()}}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	session, err := b.Execute(ctx, "say hello", ExecOptions{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}

	var gotText bool
	for msg := range session.Messages {
		if msg.Type == MessageText && msg.Content == "Hello from codebuddy" {
			gotText = true
		}
	}
	if !gotText {
		t.Fatal("expected text message 'Hello from codebuddy'")
	}

	select {
	case result, ok := <-session.Result:
		if !ok {
			t.Fatal("result channel closed without a value")
		}
		if result.Status != "completed" {
			t.Fatalf("expected status=completed, got %q (error=%q)", result.Status, result.Error)
		}
		if result.Output != "Hello from codebuddy" {
			t.Fatalf("expected output 'Hello from codebuddy', got %q", result.Output)
		}
		if result.SessionID != "ses_cb_001" {
			t.Fatalf("expected session_id=ses_cb_001, got %q", result.SessionID)
		}
		// Usage is reported under the requested model (or "unknown" if none).
		if len(result.Usage) == 0 {
			t.Fatal("expected non-empty usage map")
		}
		u, ok := result.Usage["unknown"]
		if !ok {
			t.Fatalf("expected usage under 'unknown' (no model requested), got %#v", result.Usage)
		}
		if u.InputTokens != 100 || u.OutputTokens != 50 {
			t.Fatalf("unexpected usage: %+v", u)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for result")
	}
}

func TestCodebuddyExecute_NotFound(t *testing.T) {
	t.Parallel()

	b := &codebuddyBackend{cfg: Config{ExecutablePath: "/nonexistent/path/codebuddy", Logger: slog.Default()}}

	ctx := context.Background()
	_, err := b.Execute(ctx, "prompt", ExecOptions{})
	if err == nil {
		t.Fatal("expected error for missing executable")
	}
	if !strings.Contains(err.Error(), "codebuddy executable not found") {
		t.Fatalf("expected 'codebuddy executable not found' in error, got %q", err.Error())
	}
}

// fakeCodebuddyACPSetModelFailureScript acks initialize / session/new
// but rejects session/set_model with a JSON-RPC error — the scenario
// codebuddyBackend must propagate as a failed task rather than silently
// falling back to the default model.
func fakeCodebuddyACPSetModelFailureScript() string {
	return `#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":1,"agentCapabilities":{}}}\n' "$id"
      ;;
    *'"method":"session/new"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"sessionId":"ses_cb_002"}}\n' "$id"
      ;;
    *'"method":"session/set_model"'*)
      printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32602,"message":"model not available: bogus-model"}}\n' "$id"
      exit 0
      ;;
  esac
done
`
}

func TestCodebuddyExecute_SetModelFailureFailsTask(t *testing.T) {
	t.Parallel()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fixture is POSIX-only")
	}

	fakePath := filepath.Join(t.TempDir(), "codebuddy")
	writeTestExecutable(t, fakePath, []byte(fakeCodebuddyACPSetModelFailureScript()))

	b := &codebuddyBackend{cfg: Config{ExecutablePath: fakePath, Logger: slog.Default()}}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	session, err := b.Execute(ctx, "prompt-ignored", ExecOptions{
		Model:   "bogus-model",
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	go func() {
		for range session.Messages {
		}
	}()

	select {
	case result, ok := <-session.Result:
		if !ok {
			t.Fatal("result channel closed without a value")
		}
		if result.Status != "failed" {
			t.Fatalf("expected status=failed, got %q (error=%q)", result.Status, result.Error)
		}
		if !strings.Contains(result.Error, `could not switch to model "bogus-model"`) {
			t.Errorf("expected error to name the requested model, got %q", result.Error)
		}
		if !strings.Contains(result.Error, "model not available") {
			t.Errorf("expected error to surface upstream message, got %q", result.Error)
		}
		if result.SessionID != "ses_cb_002" {
			t.Errorf("expected session id preserved on failure, got %q", result.SessionID)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for result")
	}
}

// fakeCodebuddyACPResumeScript acks initialize + session/resume (echoes
// the requested sessionId) + session/set_model + session/prompt so the
// resume path completes cleanly. Records all inbound frames to
// $CODEBUDDY_FRAMES so tests can assert session/resume was invoked.
func fakeCodebuddyACPResumeScript(recordPath string) string {
	return `#!/bin/sh
RECORD_PATH=` + recordPath + `
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$RECORD_PATH"
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":1,"agentCapabilities":{}}}\n' "$id"
      ;;
    *'"method":"session/resume"'*)
      sid=$(printf '%s' "$line" | sed -n 's/.*"sessionId":"\([^"]*\)".*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"sessionId":"%s"}}\n' "$id" "$sid"
      ;;
    *'"method":"session/set_model"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      ;;
    *'"method":"session/prompt"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"end_turn"}}\n' "$id"
      exit 0
      ;;
  esac
done
`
}

func TestCodebuddyExecute_Resume(t *testing.T) {
	t.Parallel()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fixture is POSIX-only")
	}

	recordPath := filepath.Join(t.TempDir(), "frames.jsonl")
	fakePath := filepath.Join(t.TempDir(), "codebuddy")
	writeTestExecutable(t, fakePath, []byte(fakeCodebuddyACPResumeScript(recordPath)))

	b := &codebuddyBackend{cfg: Config{ExecutablePath: fakePath, Logger: slog.Default()}}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	session, err := b.Execute(ctx, "prompt-ignored", ExecOptions{
		Timeout:         5 * time.Second,
		ResumeSessionID: "ses_resume_me",
		Model:           "claude-sonnet-4.6",
	})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	go func() {
		for range session.Messages {
		}
	}()
	select {
	case result, ok := <-session.Result:
		if !ok {
			t.Fatal("result channel closed without a value")
		}
		if result.Status != "completed" {
			t.Fatalf("expected status=completed, got %q (error=%q)", result.Status, result.Error)
		}
		if result.SessionID != "ses_resume_me" {
			t.Fatalf("expected session_id=ses_resume_me, got %q", result.SessionID)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for result")
	}

	// session/resume must have been invoked (not session/new).
	frame := findRecordedFrame(t, recordPath, "session/resume")
	params := frame["params"].(map[string]any)
	if params["sessionId"] != "ses_resume_me" {
		t.Fatalf("session/resume sessionId = %v, want ses_resume_me", params["sessionId"])
	}
	// session/new must NOT have been recorded.
	if hasFrame(t, recordPath, "session/new") {
		t.Fatal("session/new should not be invoked when ResumeSessionID is set")
	}
}

// hasFrame reports whether a recorded frame for the given method exists.
func hasFrame(t *testing.T, recordPath, method string) bool {
	t.Helper()
	data, err := os.ReadFile(recordPath)
	if err != nil {
		t.Fatalf("read record file: %v", err)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var frame map[string]any
		if err := json.Unmarshal([]byte(line), &frame); err != nil {
			continue
		}
		if frame["method"] == method {
			return true
		}
	}
	return false
}

func TestParseCodebuddyModels_FullHelp(t *testing.T) {
	t.Parallel()
	helpOutput := `Usage: codebuddy [options] [command] [prompt]

Options:
  --model <model>                                  Model for the current session. Please provide the model ID. Currently supported: (claude-sonnet-4.6, claude-opus-4.7, gemini-3.1-pro, gpt-5.5, glm-5.1-ioa, minimax-m2.7-ioa, kimi-k2.6-ioa, hy3-preview-ioa, deepseek-v3-2-volc-ioa)
  --effort <level>                                 Reasoning effort level (low, medium, high, xhigh)
`
	models := parseCodebuddyModels(helpOutput)
	if len(models) != 9 {
		t.Fatalf("expected 9 models, got %d: %+v", len(models), models)
	}
	if !models[0].Default {
		t.Error("first model should be marked as default")
	}
	if models[0].ID != "claude-sonnet-4.6" {
		t.Errorf("first model ID = %q, want claude-sonnet-4.6", models[0].ID)
	}
	if models[0].Provider != "anthropic" {
		t.Errorf("claude model provider = %q, want anthropic", models[0].Provider)
	}
	// Spot check providers
	providers := map[string]string{}
	for _, m := range models {
		providers[m.ID] = m.Provider
	}
	checks := map[string]string{
		"gpt-5.5":                "openai",
		"gemini-3.1-pro":         "google",
		"glm-5.1-ioa":            "zhipu",
		"minimax-m2.7-ioa":       "minimax",
		"kimi-k2.6-ioa":          "kimi",
		"hy3-preview-ioa":        "hunyuan",
		"deepseek-v3-2-volc-ioa": "deepseek",
	}
	for id, want := range checks {
		if got := providers[id]; got != want {
			t.Errorf("provider(%q) = %q, want %q", id, got, want)
		}
	}
}

func TestParseCodebuddyModels_Malformed(t *testing.T) {
	t.Parallel()
	models := parseCodebuddyModels("totally unrelated output\nno model line here")
	if len(models) != 0 {
		t.Fatalf("expected 0 models from malformed output, got %d", len(models))
	}
}

func TestParseCodebuddyEffortHelp(t *testing.T) {
	t.Parallel()
	helpOutput := `  --effort <level>                                 Reasoning effort level (low, medium, high, xhigh)`
	levels := parseCodebuddyEffortHelp(helpOutput)
	expected := []string{"low", "medium", "high", "xhigh"}
	if len(levels) != len(expected) {
		t.Fatalf("expected %d levels, got %d: %v", len(expected), len(levels), levels)
	}
	for i, l := range levels {
		if l != expected[i] {
			t.Errorf("level[%d]: expected %q, got %q", i, expected[i], l)
		}
	}
}

func TestParseCodebuddyEffortHelp_Missing(t *testing.T) {
	t.Parallel()
	levels := parseCodebuddyEffortHelp("no effort line here")
	if len(levels) != 0 {
		t.Fatalf("expected nil for missing effort line, got %v", levels)
	}
}

func TestIsKnownThinkingValue_Codebuddy(t *testing.T) {
	t.Parallel()
	cases := []struct {
		value string
		want  bool
	}{
		{"", true},
		{"low", true},
		{"medium", true},
		{"high", true},
		{"xhigh", true},
		{"max", false},
		{"none", false},
	}
	for _, tc := range cases {
		got := IsKnownThinkingValue("codebuddy", tc.value)
		if got != tc.want {
			t.Errorf("IsKnownThinkingValue(codebuddy, %q) = %v, want %v", tc.value, got, tc.want)
		}
	}
}
