package streaming

import (
	"net/http"

	"gpt-load/internal/models"
)

// ChannelRetryFunc defines the function signature for retry requests
type ChannelRetryFunc func(accumulatedText string) (*http.Response, error)

// StreamProcessor defines the interface for stream processing
type StreamProcessor interface {
	// HandleStreamingResponse handles streaming response with retry logic
	HandleStreamingResponse(
		resp *http.Response,
		writer http.ResponseWriter,
		group *models.Group,
		channelType string,
		originalRequest interface{},
		retryFunc ChannelRetryFunc,
	) error

	// GetStreamConfig returns the stream configuration for this processor
	GetStreamConfig() StreamConfig
}

// DefaultStreamProcessor is the default implementation of StreamProcessor
type DefaultStreamProcessor struct {
	handler *StreamHandler
	config  StreamConfig
}

// NewDefaultStreamProcessor creates a new default stream processor
func NewDefaultStreamProcessor(config StreamConfig) *DefaultStreamProcessor {
	return &DefaultStreamProcessor{
		handler: NewStreamHandler(config),
		config:  config,
	}
}

// HandleStreamingResponse implements StreamProcessor interface
func (p *DefaultStreamProcessor) HandleStreamingResponse(
	resp *http.Response,
	writer http.ResponseWriter,
	group *models.Group,
	channelType string,
	originalRequest interface{},
	retryFunc ChannelRetryFunc,
) error {
	return p.handler.HandleStreamingResponse(resp, writer, channelType, originalRequest, retryFunc)
}

// GetStreamConfig implements StreamProcessor interface
func (p *DefaultStreamProcessor) GetStreamConfig() StreamConfig {
	return p.config
}

// StreamProcessorFactory creates stream processors for different channels
type StreamProcessorFactory struct{}

// NewStreamProcessorFactory creates a new stream processor factory
func NewStreamProcessorFactory() *StreamProcessorFactory {
	return &StreamProcessorFactory{}
}

// CreateProcessor creates a stream processor for the given channel type and group
func (f *StreamProcessorFactory) CreateProcessor(channelType string, group *models.Group) StreamProcessor {
	// Base configuration
	config := StreamConfig{
		MaxRetries:                3,
		RetryDelay:                1 * 1000 * 1000 * 1000, // 1 second in nanoseconds
		EnablePunctuationHeuristic: true,
		DoneTokenPatterns:         []string{"[done]", "[DONE]", "done", "DONE"},
	}

	// Channel-specific configurations
	switch channelType {
	case "gemini":
		config.MaxRetries = 5 // Gemini is more prone to forgetting [done]
		config.DoneTokenPatterns = []string{"[done]", "[DONE]", "done", "DONE"}
		config.EnablePunctuationHeuristic = true
		
	case "openai":
		config.MaxRetries = 2 // OpenAI is more reliable
		config.DoneTokenPatterns = []string{} // OpenAI uses [DONE] signal
		config.EnablePunctuationHeuristic = false
		
	case "anthropic":
		config.MaxRetries = 2
		config.DoneTokenPatterns = []string{} // Anthropic uses message_stop signal
		config.EnablePunctuationHeuristic = false
		
	default:
		// Generic configuration for unknown channels
		config.MaxRetries = 3
		config.DoneTokenPatterns = []string{"[done]", "[DONE]", "done", "DONE"}
		config.EnablePunctuationHeuristic = true
	}

	return NewDefaultStreamProcessor(config)
}