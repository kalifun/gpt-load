package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/models"
	"gpt-load/internal/utils"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

func (ps *ProxyServer) handleStreamingResponse(c *gin.Context, resp *http.Response, channelHandler channel.ChannelProxy, group *models.Group, bodyBytes []byte) {
	// Check if this channel type should use simple proxy mode
	channelType := channelHandler.GetChannelType()
	
	// For OpenAI and Anthropic, use simple proxy mode (direct streaming)
	// Only Gemini uses intelligent streaming with retry logic
	if channelType == "openai" || channelType == "anthropic" {
		ps.handleSimpleStreamingResponse(c, resp)
		return
	}
	
	// For Gemini and other channels, use intelligent streaming with retry logic
	processor := ps.streamProcessorFactory.CreateProcessor(channelType, group)

	// Create retry function that can make new requests with accumulated context
	retryFunc := func(accumulatedText string) (*http.Response, error) {
		return ps.createRetryRequest(c, channelHandler, group, bodyBytes, accumulatedText)
	}

	// Handle the streaming response with retry logic
	err := processor.HandleStreamingResponse(resp, c.Writer, group, channelType, bodyBytes, retryFunc)
	if err != nil {
		logrus.Errorf("Intelligent streaming response handling failed: %v", err)
		// If intelligent streaming fails, try to fall back to simple streaming
		ps.handleSimpleStreamingResponse(c, resp)
	}
}

// createRetryRequest creates a new request for retry with accumulated context
func (ps *ProxyServer) createRetryRequest(
	c *gin.Context,
	channelHandler channel.ChannelProxy,
	group *models.Group,
	originalBodyBytes []byte,
	accumulatedText string,
) (*http.Response, error) {
	// Parse original request body
	var originalBody map[string]interface{}
	if err := json.Unmarshal(originalBodyBytes, &originalBody); err != nil {
		return nil, fmt.Errorf("failed to parse original request body: %w", err)
	}

	// Build retry request body with accumulated context
	retryBody := ps.buildRetryRequestBody(originalBody, accumulatedText, channelHandler.GetChannelType())

	// Marshal retry body
	retryBodyBytes, err := json.Marshal(retryBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal retry body: %w", err)
	}

	// Get API key for retry
	apiKey, err := ps.keyProvider.SelectKey(group.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to get API key for retry: %w", err)
	}

	// Build upstream URL
	upstreamURL, err := channelHandler.BuildUpstreamURL(c.Request.URL, group)
	if err != nil {
		return nil, fmt.Errorf("failed to build upstream URL: %w", err)
	}

	// Create retry request
	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, c.Request.Method, upstreamURL, bytes.NewReader(retryBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create retry request: %w", err)
	}

	req.ContentLength = int64(len(retryBodyBytes))
	req.Header = c.Request.Header.Clone()

	// Clean up client auth keys
	req.Header.Del("Authorization")
	req.Header.Del("X-Api-Key")
	req.Header.Del("X-Goog-Api-Key")

	// Apply custom header rules
	if len(group.HeaderRuleList) > 0 {
		headerCtx := utils.NewHeaderVariableContextFromGin(c, group, apiKey)
		utils.ApplyHeaderRules(req, group.HeaderRuleList, headerCtx)
	}

	// Apply channel-specific modifications
	channelHandler.ModifyRequest(req, apiKey, group)

	// Get appropriate client
	client := channelHandler.GetStreamClient()
	channelHandler.ReshapeStreamReqBody(req)
	req.Header.Set("X-Accel-Buffering", "no")

	// Make the request
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("retry request failed: %w", err)
	}

	return resp, nil
}

// buildRetryRequestBody builds a retry request body with accumulated context
func (ps *ProxyServer) buildRetryRequestBody(
	originalBody map[string]interface{},
	accumulatedText string,
	channelType string,
) map[string]interface{} {
	retryBody := make(map[string]interface{})

	// Copy all original fields
	for k, v := range originalBody {
		retryBody[k] = v
	}

	// Add retry context based on channel type
	switch channelType {
	case "openai":
		ps.addOpenAIRetryContext(retryBody, accumulatedText)
	case "gemini":
		ps.addGeminiRetryContext(retryBody, accumulatedText)
	case "anthropic":
		ps.addAnthropicRetryContext(retryBody, accumulatedText)
	default:
		ps.addGenericRetryContext(retryBody, accumulatedText)
	}

	return retryBody
}

// addOpenAIRetryContext adds retry context for OpenAI requests
func (ps *ProxyServer) addOpenAIRetryContext(body map[string]interface{}, accumulatedText string) {
	messages, ok := body["messages"].([]interface{})
	if !ok {
		return
	}

	// Add a system message with context
	systemMessage := map[string]interface{}{
		"role": "system",
		"content": fmt.Sprintf("Continue from where you left off. Previous response: %s\n\nContinue generating the response without repetition.", accumulatedText),
	}

	// Insert at the beginning
	newMessages := append([]interface{}{systemMessage}, messages...)
	body["messages"] = newMessages
}

// addGeminiRetryContext adds retry context for Gemini requests
func (ps *ProxyServer) addGeminiRetryContext(body map[string]interface{}, accumulatedText string) {
	contents, ok := body["contents"].([]interface{})
	if !ok {
		return
	}

	// Find the last user message to insert context after it
	lastUserIndex := -1
	for i := len(contents) - 1; i >= 0; i-- {
		if content, ok := contents[i].(map[string]interface{}); ok {
			if role, ok := content["role"].(string); ok && role == "user" {
				lastUserIndex = i
				break
			}
		}
	}

	// Create context messages
	contextMessages := []interface{}{
		map[string]interface{}{
			"role": "model",
			"parts": []interface{}{
				map[string]interface{}{"text": accumulatedText},
			},
		},
		map[string]interface{}{
			"role": "user",
			"parts": []interface{}{
				map[string]interface{}{"text": "Continue exactly where you left off without any preamble or repetition. Remember to include [done] at the end."},
			},
		},
	}

	// Insert context after last user message
	if lastUserIndex != -1 {
		newContents := make([]interface{}, 0, len(contents)+2)
		newContents = append(newContents, contents[:lastUserIndex+1]...)
		newContents = append(newContents, contextMessages...)
		newContents = append(newContents, contents[lastUserIndex+1:]...)
		body["contents"] = newContents
	} else {
		body["contents"] = append(contents, contextMessages...)
	}
}

// addAnthropicRetryContext adds retry context for Anthropic requests
func (ps *ProxyServer) addAnthropicRetryContext(body map[string]interface{}, accumulatedText string) {
	// For Anthropic, we typically need to modify the messages array
	messages, ok := body["messages"].([]interface{})
	if !ok {
		return
	}

	// Add context as a user message
	contextMessage := map[string]interface{}{
		"role": "user",
		"content": fmt.Sprintf("Continue from where you left off. Previous response: %s\n\nContinue without repetition.", accumulatedText),
	}

	body["messages"] = append(messages, contextMessage)
}

// addGenericRetryContext adds retry context for generic requests
func (ps *ProxyServer) addGenericRetryContext(body map[string]interface{}, accumulatedText string) {
	// Generic fallback - try to add context to messages if available
	if messages, ok := body["messages"].([]interface{}); ok {
		contextMessage := map[string]interface{}{
			"role": "user",
			"content": fmt.Sprintf("Continue from where you left off. Previous response: %s\n\nContinue without repetition.", accumulatedText),
		}
		body["messages"] = append(messages, contextMessage)
	}
}

// handleSimpleStreamingResponse handles streaming response with simple proxy mode (direct streaming)
func (ps *ProxyServer) handleSimpleStreamingResponse(c *gin.Context, resp *http.Response) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		logrus.Error("Streaming unsupported by the writer, falling back to normal response")
		ps.handleNormalResponse(c, resp)
		return
	}

	buf := make([]byte, 4*1024)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := c.Writer.Write(buf[:n]); writeErr != nil {
				logUpstreamError("writing stream to client", writeErr)
				return
			}
			flusher.Flush()
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			logUpstreamError("reading from upstream", err)
			return
		}
	}
}

func (ps *ProxyServer) handleNormalResponse(c *gin.Context, resp *http.Response) {
	if _, err := io.Copy(c.Writer, resp.Body); err != nil {
		logUpstreamError("copying response body", err)
	}
}
