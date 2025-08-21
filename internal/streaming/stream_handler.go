package streaming

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// StreamHandler handles streaming responses with intelligent retry logic
type StreamHandler struct {
	maxRetries           int
	retryDelay           time.Duration
	enablePunctuationHeuristic bool
	doneTokenPatterns    []string
}

// StreamConfig configures the streaming handler
type StreamConfig struct {
	MaxRetries                int
	RetryDelay                time.Duration
	EnablePunctuationHeuristic bool
	DoneTokenPatterns         []string
}

// NewStreamHandler creates a new streaming handler
func NewStreamHandler(config StreamConfig) *StreamHandler {
	if config.MaxRetries <= 0 {
		config.MaxRetries = 3
	}
	if config.RetryDelay <= 0 {
		config.RetryDelay = 1 * time.Second
	}
	if len(config.DoneTokenPatterns) == 0 {
		config.DoneTokenPatterns = []string{"[done]", "[DONE]", "done", "DONE"}
	}

	return &StreamHandler{
		maxRetries:           config.MaxRetries,
		retryDelay:           config.RetryDelay,
		enablePunctuationHeuristic: config.EnablePunctuationHeuristic,
		doneTokenPatterns:    config.DoneTokenPatterns,
	}
}

// HandleStreamingResponse handles streaming response with retry logic
func (sh *StreamHandler) HandleStreamingResponse(
	resp *http.Response,
	writer http.ResponseWriter,
	channelType string,
	originalRequest interface{},
	retryRequestFunc func(accumulatedText string) (*http.Response, error),
) error {
	var accumulatedText string
	consecutiveRetryCount := 0
	resumePunctStreak := 0

	for {
		logrus.Debugf("=== Starting stream attempt %d/%d ===", consecutiveRetryCount+1, sh.maxRetries+1)

		cleanExit, err := sh.processStreamAttempt(
			resp, writer, channelType, &accumulatedText,
			&resumePunctStreak, consecutiveRetryCount,
		)

		if err != nil {
			return err
		}

		if cleanExit {
			logrus.Info("=== STREAM COMPLETED SUCCESSFULLY ===")
			return nil
		}

		// Check if we've exceeded max retries
		if consecutiveRetryCount >= sh.maxRetries {
			return sh.writeRetryError(writer, consecutiveRetryCount)
		}

		// Prepare for retry
		consecutiveRetryCount++
		logrus.Infof("=== STARTING RETRY %d/%d ===", consecutiveRetryCount, sh.maxRetries)

		// Close current response body
		resp.Body.Close()

		// Make retry request
		time.Sleep(sh.retryDelay)
		newResp, err := retryRequestFunc(accumulatedText)
		if err != nil {
			logrus.Errorf("Retry request failed: %v", err)
			return err
		}

		resp = newResp
	}
}

// processStreamAttempt processes a single stream attempt
func (sh *StreamHandler) processStreamAttempt(
	resp *http.Response,
	writer http.ResponseWriter,
	channelType string,
	accumulatedText *string,
	resumePunctStreak *int,
	attempt int,
) (bool, error) {
	// Set streaming headers
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "no-cache")
	writer.Header().Set("Connection", "keep-alive")
	writer.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := writer.(http.Flusher)
	if !ok {
		return false, fmt.Errorf("streaming not supported")
	}

	scanner := bufio.NewScanner(resp.Body)
	var lastTextChunk string
	var textInThisStream string

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		// Parse SSE line
		if strings.HasPrefix(line, "data: ") {
			dataContent := strings.TrimPrefix(line, "data: ")
			if dataContent == "[DONE]" {
				// OpenAI style end
				logrus.Debug("Received [DONE] signal")
				return true, nil
			}

			// Parse JSON data
			var data map[string]interface{}
			if err := json.Unmarshal([]byte(dataContent), &data); err != nil {
				logrus.Debugf("Failed to parse JSON data: %v", err)
				continue
			}

			// Extract text based on channel type
			textChunk := sh.extractTextFromData(data, channelType)
			if textChunk != "" {
				lastTextChunk = textChunk
				*accumulatedText += textChunk
				textInThisStream += textChunk
			}

			// Forward the line to client, but remove [done] tokens for Gemini
			processedLine := line
			if channelType == "gemini" {
				processedLine = sh.removeDoneTokensFromLine(line, data)
			}
			
			if _, err := fmt.Fprintf(writer, "%s\n\n", processedLine); err != nil {
				return false, fmt.Errorf("failed to write to client: %w", err)
			}
			flusher.Flush()

			// Check for completion
			if sh.isStreamComplete(data, channelType, *accumulatedText) {
				return true, nil
			}
		} else {
			// Forward non-data lines as-is
			if _, err := fmt.Fprintf(writer, "%s\n\n", line); err != nil {
				return false, fmt.Errorf("failed to write to client: %w", err)
			}
			flusher.Flush()
		}
	}

	// Check for stream completion without explicit end signal
	if err := scanner.Err(); err != nil {
		logrus.Errorf("Stream error: %v", err)
		return false, nil // Trigger retry
	}

	// Stream ended without explicit completion signal
	logrus.Debug("Stream ended without explicit completion signal")

	// Apply punctuation heuristic for resumed attempts
	if sh.enablePunctuationHeuristic && attempt > 0 && sh.endsWithSentencePunctuation(lastTextChunk) {
		*resumePunctStreak++
		logrus.Debugf("Resume punctuation streak: %d", *resumePunctStreak)
		if *resumePunctStreak >= 3 {
			logrus.Info("Stream completed due to punctuation heuristic")
			return true, nil
		}
	} else {
		*resumePunctStreak = 0
	}

	// Check if we have any content and it seems complete
	if sh.isContentComplete(*accumulatedText, channelType) {
		logrus.Info("Stream completed based on content analysis")
		return true, nil
	}

	// Trigger retry
	return false, nil
}

// extractTextFromData extracts text from streaming data based on channel type
func (sh *StreamHandler) extractTextFromData(data map[string]interface{}, channelType string) string {
	switch channelType {
	case "openai":
		return sh.extractOpenAIText(data)
	case "gemini":
		return sh.extractGeminiText(data)
	case "anthropic":
		return sh.extractAnthropicText(data)
	default:
		return sh.extractGenericText(data)
	}
}

// extractOpenAIText extracts text from OpenAI streaming format
func (sh *StreamHandler) extractOpenAIText(data map[string]interface{}) string {
	choices, ok := data["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return ""
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return ""
	}

	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		return ""
	}

	content, ok := delta["content"].(string)
	if ok {
		return content
	}

	return ""
}

// extractGeminiText extracts text from Gemini streaming format
func (sh *StreamHandler) extractGeminiText(data map[string]interface{}) string {
	candidates, ok := data["candidates"].([]interface{})
	if !ok || len(candidates) == 0 {
		return ""
	}

	candidate, ok := candidates[0].(map[string]interface{})
	if !ok {
		return ""
	}

	content, ok := candidate["content"].(map[string]interface{})
	if !ok {
		return ""
	}

	parts, ok := content["parts"].([]interface{})
	if !ok || len(parts) == 0 {
		return ""
	}

	part, ok := parts[0].(map[string]interface{})
	if !ok {
		return ""
	}

	text, ok := part["text"].(string)
	if ok {
		return text
	}

	return ""
}

// extractAnthropicText extracts text from Anthropic streaming format
func (sh *StreamHandler) extractAnthropicText(data map[string]interface{}) string {
	if typ, ok := data["type"].(string); ok && typ == "content_block_delta" {
		if delta, ok := data["delta"].(map[string]interface{}); ok {
			if text, ok := delta["text"].(string); ok {
				return text
			}
		}
	}
	return ""
}

// extractGenericText extracts text from generic format
func (sh *StreamHandler) extractGenericText(data map[string]interface{}) string {
	if text, ok := data["text"].(string); ok {
		return text
	}
	if content, ok := data["content"].(string); ok {
		return content
	}
	return ""
}

// isStreamComplete checks if the stream is complete based on channel-specific signals
func (sh *StreamHandler) isStreamComplete(data map[string]interface{}, channelType string, accumulatedText string) bool {
	switch channelType {
	case "openai":
		return sh.isOpenAIComplete(data)
	case "gemini":
		return sh.isGeminiComplete(data, accumulatedText)
	case "anthropic":
		return sh.isAnthropicComplete(data)
	default:
		return sh.isGenericComplete(data, accumulatedText)
	}
}

// isOpenAIComplete checks if OpenAI stream is complete
func (sh *StreamHandler) isOpenAIComplete(data map[string]interface{}) bool {
	choices, ok := data["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return false
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return false
	}

	finishReason, ok := choice["finish_reason"].(string)
	if ok && (finishReason == "stop" || finishReason == "length") {
		return true
	}

	return false
}

// isGeminiComplete checks if Gemini stream is complete
func (sh *StreamHandler) isGeminiComplete(data map[string]interface{}, accumulatedText string) bool {
	// Check for [done] token in accumulated text
	for _, pattern := range sh.doneTokenPatterns {
		if strings.Contains(accumulatedText, pattern) {
			return true
		}
	}

	// Check for finish reason in metadata
	if metadata, ok := data["metadata"].(map[string]interface{}); ok {
		if finishReason, ok := metadata["finishReason"].(string); ok && finishReason == "STOP" {
			return true
		}
	}

	return false
}

// isAnthropicComplete checks if Anthropic stream is complete
func (sh *StreamHandler) isAnthropicComplete(data map[string]interface{}) bool {
	if typ, ok := data["type"].(string); ok && typ == "message_stop" {
		return true
	}
	return false
}

// isGenericComplete checks if generic stream is complete
func (sh *StreamHandler) isGenericComplete(data map[string]interface{}, accumulatedText string) bool {
	// Check for [done] token in accumulated text
	for _, pattern := range sh.doneTokenPatterns {
		if strings.Contains(accumulatedText, pattern) {
			return true
		}
	}

	// Check for finish reason
	if finishReason, ok := data["finish_reason"].(string); ok {
		if finishReason == "stop" || finishReason == "length" {
			return true
		}
	}

	return false
}

// isContentComplete checks if content appears complete based on heuristics
func (sh *StreamHandler) isContentComplete(text string, channelType string) bool {
	if text == "" {
		return false
	}

	// For Gemini, specifically check for [done] token
	if channelType == "gemini" {
		for _, pattern := range sh.doneTokenPatterns {
			if strings.Contains(text, pattern) {
				return true
			}
		}
	}

	// Generic completion check
	return sh.endsWithSentencePunctuation(text) && len(text) > 50
}

// endsWithSentencePunctuation checks if text ends with sentence punctuation
func (sh *StreamHandler) endsWithSentencePunctuation(text string) bool {
	trimmed := strings.TrimSpace(text)
	if len(trimmed) == 0 {
		return false
	}
	
	runes := []rune(trimmed)
	last := runes[len(runes)-1]
	const punctuations = "。？！.!?…\"'\"'"
	return strings.ContainsRune(punctuations, last)
}

// removeDoneTokensFromLine removes [done] tokens from Gemini streaming responses
func (sh *StreamHandler) removeDoneTokensFromLine(line string, data map[string]interface{}) string {
	if !strings.HasPrefix(line, "data: ") {
		return line
	}
	
	dataContent := strings.TrimPrefix(line, "data: ")
	if dataContent == "[DONE]" {
		return line // OpenAI style [DONE] should be preserved
	}
	
	// Parse JSON data
	var parsedData map[string]interface{}
	if err := json.Unmarshal([]byte(dataContent), &parsedData); err != nil {
		return line
	}
	
	// Extract text from Gemini format
	text := sh.extractGeminiText(parsedData)
	if text == "" {
		return line
	}
	
	// Remove [done] tokens from text
	cleanText := sh.RemoveDoneTokensFromText(text)
	
	// If text was modified, reconstruct the JSON
	if cleanText != text {
		// Update the text in the parsed data
		if candidates, ok := parsedData["candidates"].([]interface{}); ok && len(candidates) > 0 {
			if candidate, ok := candidates[0].(map[string]interface{}); ok {
				if content, ok := candidate["content"].(map[string]interface{}); ok {
					if parts, ok := content["parts"].([]interface{}); ok && len(parts) > 0 {
						if part, ok := parts[0].(map[string]interface{}); ok {
							part["text"] = cleanText
						}
					}
				}
			}
		}
		
		// Marshal back to JSON
		newDataBytes, err := json.Marshal(parsedData)
		if err == nil {
			return "data: " + string(newDataBytes)
		}
	}
	
	return line
}

// RemoveDoneTokensFromText removes [done] tokens from text
func (sh *StreamHandler) RemoveDoneTokensFromText(text string) string {
	// Remove [done] tokens from the end of text
	for _, pattern := range sh.doneTokenPatterns {
		if strings.HasSuffix(text, pattern) {
			text = strings.TrimSuffix(text, pattern)
			// Also remove any whitespace before the token
			text = strings.TrimSpace(text)
			break
		}
	}
	return text
}

// writeRetryError writes a retry error to the client
func (sh *StreamHandler) writeRetryError(writer http.ResponseWriter, retryCount int) error {
	errorPayload := map[string]interface{}{
		"error": map[string]interface{}{
			"code":    504,
			"status":  "DEADLINE_EXCEEDED",
			"message": fmt.Sprintf("Retry limit (%d) exceeded after stream interruption", sh.maxRetries),
		},
	}

	errorBytes, _ := json.Marshal(errorPayload)
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(504)
	
	if _, err := writer.Write(errorBytes); err != nil {
		return fmt.Errorf("failed to write error response: %w", err)
	}

	return fmt.Errorf("retry limit exceeded")
}