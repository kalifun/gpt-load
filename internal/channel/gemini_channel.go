package channel

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	app_errors "gpt-load/internal/errors"
	"gpt-load/internal/models"
	"gpt-load/internal/utils"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

func init() {
	Register("gemini", newGeminiChannel)
}

type GeminiChannel struct {
	*BaseChannel
}

func newGeminiChannel(f *Factory, group *models.Group) (ChannelProxy, error) {
	base, err := f.newBaseChannel("gemini", group)
	if err != nil {
		return nil, err
	}

	return &GeminiChannel{
		BaseChannel: base,
	}, nil
}

// ModifyRequest adds the API key as a query parameter for Gemini requests.
func (ch *GeminiChannel) ModifyRequest(req *http.Request, apiKey *models.APIKey, group *models.Group) {
	if strings.Contains(req.URL.Path, "v1beta/openai") {
		req.Header.Set("Authorization", "Bearer "+apiKey.KeyValue)
	} else {
		q := req.URL.Query()
		q.Set("key", apiKey.KeyValue)
		req.URL.RawQuery = q.Encode()
	}
}

// IsStreamRequest checks if the request is for a streaming response.
func (ch *GeminiChannel) IsStreamRequest(c *gin.Context, bodyBytes []byte) bool {
	path := c.Request.URL.Path
	if strings.HasSuffix(path, ":streamGenerateContent") {
		return true
	}

	// Also check for standard streaming indicators as a fallback.
	if strings.Contains(c.GetHeader("Accept"), "text/event-stream") {
		return true
	}
	if c.Query("stream") == "true" {
		return true
	}

	type streamPayload struct {
		Stream bool `json:"stream"`
	}
	var p streamPayload
	if err := json.Unmarshal(bodyBytes, &p); err == nil {
		return p.Stream
	}

	return false
}

func (ch *GeminiChannel) ExtractModel(c *gin.Context, bodyBytes []byte) string {
	// gemini format
	path := c.Request.URL.Path
	parts := strings.Split(path, "/")
	for i, part := range parts {
		if part == "models" && i+1 < len(parts) {
			modelPart := parts[i+1]
			return strings.Split(modelPart, ":")[0]
		}
	}

	// openai format
	type modelPayload struct {
		Model string `json:"model"`
	}
	var p modelPayload
	if err := json.Unmarshal(bodyBytes, &p); err == nil && p.Model != "" {
		return strings.TrimPrefix(p.Model, "models/")
	}

	return ""
}

// ValidateKey checks if the given API key is valid by making a generateContent request.
func (ch *GeminiChannel) ValidateKey(ctx context.Context, apiKey *models.APIKey, group *models.Group) (bool, error) {
	upstreamURL := ch.getUpstreamURL()
	if upstreamURL == nil {
		return false, fmt.Errorf("no upstream URL configured for channel %s", ch.Name)
	}

	// Safely join the path segments
	reqURL, err := url.JoinPath(upstreamURL.String(), "v1beta", "models", ch.TestModel+":generateContent")
	if err != nil {
		return false, fmt.Errorf("failed to create gemini validation path: %w", err)
	}
	reqURL += "?key=" + apiKey.KeyValue

	payload := gin.H{
		"contents": []gin.H{
			{"parts": []gin.H{
				{"text": "hi"},
			}},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return false, fmt.Errorf("failed to marshal validation payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", reqURL, bytes.NewBuffer(body))
	if err != nil {
		return false, fmt.Errorf("failed to create validation request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Apply custom header rules if available
	if len(group.HeaderRuleList) > 0 {
		headerCtx := utils.NewHeaderVariableContext(group, apiKey)
		utils.ApplyHeaderRules(req, group.HeaderRuleList, headerCtx)
	}

	resp, err := ch.HTTPClient.Do(req)
	if err != nil {
		return false, fmt.Errorf("failed to send validation request: %w", err)
	}
	defer resp.Body.Close()

	// Any 2xx status code indicates the key is valid.
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return true, nil
	}

	// For non-200 responses, parse the body to provide a more specific error reason.
	errorBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, fmt.Errorf("key is invalid (status %d), but failed to read error body: %w", resp.StatusCode, err)
	}

	// Use the new parser to extract a clean error message.
	parsedError := app_errors.ParseUpstreamError(errorBody)

	return false, fmt.Errorf("[status %d] %s", resp.StatusCode, parsedError)
}

func (ch *GeminiChannel) ReshapeStreamReqBody(req *http.Request) {
    bodyBytes, err := io.ReadAll(req.Body)
    if err != nil {
				logrus.Errorf("Failed to read request body: %v", err)
        return
    }
    defer req.Body.Close()

    var data map[string]interface{}
    if err := json.Unmarshal(bodyBytes, &data); err != nil {
				logrus.Errorf("Failed to unmarshal request body: %v", err)
        return
    }

    injectSystemPrompt(data)

    newBody, err := json.Marshal(data)
    if err != nil {
				logrus.Errorf("Failed to marshal new request body: %v", err)
        return
    }

    req.Body = io.NopCloser(bytes.NewReader(newBody))
    req.ContentLength = int64(len(newBody))
}


// InjectSystemPrompt injects a system prompt to ensure the [done] token is present.
// It intelligently handles both system_instruction (snake_case) and systemInstruction (camelCase)
// by merging the content of system_instruction into systemInstruction before processing.
// systemInstruction is the officially recommended format.
func injectSystemPrompt(body map[string]interface{}) {
	newSystemPromptPart := map[string]interface{}{
		"text": "IMPORTANT: At the very end of your entire response, you must write the token [done] to signal completion. This is a mandatory technical requirement.",
	}

	// Standardize: If system_instruction exists, merge its content into systemInstruction.
	if snakeVal, snakeExists := body["system_instruction"]; snakeExists {
		// Ensure camelCase map exists
		camelMap, _ := body["systemInstruction"].(map[string]interface{})
		if camelMap == nil {
			camelMap = make(map[string]interface{})
		}

		// Ensure camelCase parts array exists
		camelParts, _ := camelMap["parts"].([]interface{})
		if camelParts == nil {
			camelParts = make([]interface{}, 0)
		}

		// If snake_case is a valid map with its own parts, prepend them to camelCase parts
		if snakeMap, snakeOk := snakeVal.(map[string]interface{}); snakeOk {
			if snakeParts, snakePartsOk := snakeMap["parts"].([]interface{}); snakePartsOk {
				camelParts = append(snakeParts, camelParts...)
			}
		}

		// Update the camelCase field with the merged parts and delete the snake_case one
		camelMap["parts"] = camelParts
		body["systemInstruction"] = camelMap
		delete(body, "system_instruction")
	}

	// --- From this point on, we only need to deal with systemInstruction ---

	// Case 1: systemInstruction field is missing or null. Create it.
	if val, exists := body["systemInstruction"]; !exists || val == nil {
		body["systemInstruction"] = map[string]interface{}{
			"parts": []interface{}{newSystemPromptPart},
		}
		return
	}

	instruction, ok := body["systemInstruction"].(map[string]interface{})
	if !ok {
		// The field exists but is of the wrong type. Overwrite it.
		body["systemInstruction"] = map[string]interface{}{
			"parts": []interface{}{newSystemPromptPart},
		}
		return
	}

	// Case 2: The instruction field exists, but its 'parts' array is missing, null, or not an array.
	parts, ok := instruction["parts"].([]interface{})
	if !ok {
		instruction["parts"] = []interface{}{newSystemPromptPart}
		return
	}

	// Case 3: The instruction field and its 'parts' array both exist. Append to the existing array.
	instruction["parts"] = append(parts, newSystemPromptPart)
}
