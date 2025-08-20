package streaming

import (
	"testing"
	"time"
	"gpt-load/internal/models"
)

func TestStreamHandlerCreation(t *testing.T) {
	config := StreamConfig{
		MaxRetries:                3,
		RetryDelay:                1 * time.Second,
		EnablePunctuationHeuristic: true,
		DoneTokenPatterns:         []string{"[done]", "[DONE]"},
	}

	handler := NewStreamHandler(config)
	if handler.maxRetries != 3 {
		t.Errorf("Expected maxRetries to be 3, got %d", handler.maxRetries)
	}
	if !handler.enablePunctuationHeuristic {
		t.Error("Expected EnablePunctuationHeuristic to be true")
	}
	if len(handler.doneTokenPatterns) != 2 {
		t.Errorf("Expected 2 done token patterns, got %d", len(handler.doneTokenPatterns))
	}
}

func TestStreamProcessorFactory(t *testing.T) {
	factory := NewStreamProcessorFactory()
	
	// Test Gemini processor
	group := &models.Group{ChannelType: "gemini"}
	processor := factory.CreateProcessor("gemini", group)
	if processor == nil {
		t.Error("Expected Gemini processor to be created")
	}
	
	config := processor.GetStreamConfig()
	if config.MaxRetries != 5 {
		t.Errorf("Expected Gemini maxRetries to be 5, got %d", config.MaxRetries)
	}
	
	// Test OpenAI processor
	group = &models.Group{ChannelType: "openai"}
	processor = factory.CreateProcessor("openai", group)
	if processor == nil {
		t.Error("Expected OpenAI processor to be created")
	}
	
	config = processor.GetStreamConfig()
	if config.MaxRetries != 2 {
		t.Errorf("Expected OpenAI maxRetries to be 2, got %d", config.MaxRetries)
	}
}

func TestEndsWithSentencePunctuation(t *testing.T) {
	handler := NewStreamHandler(StreamConfig{})
	
	tests := []struct {
		text     string
		expected bool
	}{
		{"Hello world.", true},
		{"Hello world!", true},
		{"Hello world?", true},
		{"Hello world。", true},
		{"Hello world！", true},
		{"Hello world？", true},
		{"Hello world", false},
		{"Hello world,", false},
		{"", false},
		{"Hello world\"", true},
		{"Hello world'", true},
	}
	
	for _, test := range tests {
		result := handler.endsWithSentencePunctuation(test.text)
		if result != test.expected {
			t.Errorf("For text '%s', expected %v, got %v", test.text, test.expected, result)
		}
	}
}