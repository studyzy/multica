// codebuddy_sdk.go 通过 CLI 控制请求获取 CodeBuddy 的真实模型列表。
//
// 不引入外部 SDK 依赖，将 codebuddy-sdk-go 中获取模型列表的核心机制
// （启动 CLI 子进程 → 发 initialize → 发 get_available_models 控制请求 →
// 解析响应）搬到此文件作为本地包内函数。
//
// 协议：stream-json 行式 JSON。控制请求格式：
//
//	{"type":"control_request","request_id":"<id>","request":{"subtype":"..."}}
//
// 控制响应格式：
//
//	{"type":"control_response","response":{"request_id":"<id>","subtype":"success","response":{...}}}
package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// codebuddyCLITimeout 是单次控制请求的超时时间。
const codebuddyCLITimeout = 30 * time.Second

// codebuddyCLI 封装与 codebuddy CLI 子进程的 stream-json 通信，
// 仅用于控制请求（initialize / get_available_models）。
type codebuddyCLI struct {
	cmd     *exec.Cmd
	stdin   *json.Encoder
	closeCh chan struct{}
	closeMu sync.Mutex
	closed  bool
	wg      sync.WaitGroup

	pendingMu sync.Mutex
	pending   map[string]chan map[string]any // request_id → 响应通道
	reqSeq    atomic.Int64
}

// newCodebuddyCLI 启动 codebuddy CLI 子进程。executablePath 为空使用 PATH 上的 codebuddy。
func newCodebuddyCLI(ctx context.Context, executablePath string) (*codebuddyCLI, error) {
	if executablePath == "" {
		executablePath = "codebuddy"
	}
	args := []string{
		"--input-format=stream-json",
		"--output-format=stream-json",
		"--verbose",
		"--setting-sources=none",
	}
	cmd := exec.CommandContext(ctx, executablePath, args...)
	hideAgentWindow(cmd)

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("codebuddy stdin pipe: %w", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("codebuddy stdout pipe: %w", err)
	}
	// stderr 丢弃，避免干扰 stream-json 输出
	cmd.Stderr = nil

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("codebuddy start: %w", err)
	}

	c := &codebuddyCLI{
		cmd:      cmd,
		stdin:    json.NewEncoder(stdinPipe),
		closeCh:  make(chan struct{}),
		pending:  make(map[string]chan map[string]any),
	}
	c.wg.Add(1)
	go c.readStdout(stdoutPipe)
	return c, nil
}

// readStdout 逐行读取 stdout JSON，路由 control_response 到 pending，其他消息丢弃。
func (c *codebuddyCLI) readStdout(pipe io.Reader) {
	defer c.wg.Done()

	scanner := bufio.NewScanner(pipe)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		select {
		case <-c.closeCh:
			return
		default:
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var data map[string]any
		if err := json.Unmarshal([]byte(line), &data); err != nil {
			continue
		}
		msgType, _ := data["type"].(string)
		if msgType != "control_response" {
			// 模型发现只关心控制响应，其他消息（assistant/message 等）忽略
			continue
		}
		c.routeControlResponse(data)
	}
}

// routeControlResponse 解析控制响应，通知对应的 sendControlRequest 等待者。
func (c *codebuddyCLI) routeControlResponse(data map[string]any) {
	resp, _ := data["response"].(map[string]any)
	if resp == nil {
		return
	}
	requestID, _ := resp["request_id"].(string)
	if requestID == "" {
		return
	}

	c.pendingMu.Lock()
	ch, ok := c.pending[requestID]
	if ok {
		delete(c.pending, requestID)
	}
	c.pendingMu.Unlock()
	if !ok {
		return
	}

	subtype, _ := resp["subtype"].(string)
	if subtype == "error" {
		errMsg, _ := resp["error"].(string)
		// 用 nil 响应 + error 信息：通过独立 error 通道传递更清晰，
		// 但为简化结构，这里把错误包装成含 __error 字段的 map
		select {
		case ch <- map[string]any{"__error": errMsg}:
		default:
		}
		return
	}
	inner, _ := resp["response"].(map[string]any)
	if inner == nil {
		inner = map[string]any{}
	}
	select {
	case ch <- inner:
	default:
	}
}

// sendControlRequest 发送控制请求并阻塞等待响应。
func (c *codebuddyCLI) sendControlRequest(ctx context.Context, payload map[string]any) (map[string]any, error) {
	requestID := fmt.Sprintf("req_%d", c.reqSeq.Add(1))
	respCh := make(chan map[string]any, 1)
	c.pendingMu.Lock()
	c.pending[requestID] = respCh
	c.pendingMu.Unlock()

	req := map[string]any{
		"type":       "control_request",
		"request_id": requestID,
		"request":    payload,
	}
	if err := c.stdin.Encode(req); err != nil {
		c.pendingMu.Lock()
		delete(c.pending, requestID)
		c.pendingMu.Unlock()
		return nil, fmt.Errorf("codebuddy write: %w", err)
	}

	select {
	case resp, ok := <-respCh:
		if !ok {
			return nil, fmt.Errorf("codebuddy connection closed")
		}
		if errMsg, isErr := resp["__error"].(string); isErr {
			return nil, fmt.Errorf("codebuddy control request %q failed: %s", payload["subtype"], errMsg)
		}
		return resp, nil
	case <-ctx.Done():
		c.pendingMu.Lock()
		delete(c.pending, requestID)
		c.pendingMu.Unlock()
		return nil, ctx.Err()
	case <-c.closeCh:
		return nil, fmt.Errorf("codebuddy connection closed")
	}
}

// close 终止子进程并清理资源。幂等。
func (c *codebuddyCLI) close() {
	c.closeMu.Lock()
	if c.closed {
		c.closeMu.Unlock()
		return
	}
	c.closed = true
	close(c.closeCh)
	c.closeMu.Unlock()

	if c.cmd != nil && c.cmd.Process != nil {
		_ = c.cmd.Process.Kill()
		_ = c.cmd.Wait()
	}
	c.wg.Wait()
}

// codebuddyAvailableModel 是 CLI 返回的模型条目（简化格式）。
type codebuddyAvailableModel struct {
	ModelID     string `json:"modelId"`
	DisplayName string `json:"displayName"`
}

// discoverCodebuddyModelsViaSDK 通过 CLI 控制请求获取真实模型列表。
// executablePath 为空时使用 PATH 上的默认 codebuddy。
func discoverCodebuddyModelsViaSDK(ctx context.Context, executablePath string) ([]Model, error) {
	cli, err := newCodebuddyCLI(ctx, executablePath)
	if err != nil {
		return nil, err
	}
	defer cli.close()

	// 1. initialize（等待 CLI 就绪）。超时或失败不视为致命错误，
	//    继续尝试 get_available_models。
	initCtx, cancel := context.WithTimeout(ctx, codebuddyCLITimeout)
	defer cancel()
	_, _ = cli.sendControlRequest(initCtx, map[string]any{
		"subtype":   "initialize",
		"hasPrompt": false,
		"capabilities": map[string]any{
			"askUserQuestion": true,
		},
	})

	// 2. get_available_models
	modelsCtx, cancel2 := context.WithTimeout(ctx, codebuddyCLITimeout)
	defer cancel2()
	resp, err := cli.sendControlRequest(modelsCtx, map[string]any{"subtype": "get_available_models"})
	if err != nil {
		return nil, fmt.Errorf("codebuddy get_available_models: %w", err)
	}

	rawModels, _ := resp["availableModels"].([]any)
	if len(rawModels) == 0 {
		return nil, fmt.Errorf("codebuddy returned no models")
	}
	models := make([]Model, 0, len(rawModels))
	for i, item := range rawModels {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		b, err := json.Marshal(m)
		if err != nil {
			continue
		}
		var am codebuddyAvailableModel
		if err := json.Unmarshal(b, &am); err != nil {
			continue
		}
		if am.ModelID == "" {
			continue
		}
		label := am.DisplayName
		if label == "" {
			label = codebuddyModelLabel(am.ModelID)
		}
		models = append(models, Model{
			ID:       am.ModelID,
			Label:    label,
			Provider: codebuddyModelProvider(am.ModelID),
			Default:  i == 0,
		})
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("codebuddy returned no usable models")
	}
	return models, nil
}
