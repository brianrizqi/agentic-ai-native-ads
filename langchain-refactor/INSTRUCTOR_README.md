# Instructor-based Classification for Gemma v7

This directory contains an **Instructor-based** implementation for native ads classification using the fine-tuned **Gemma v7** model (90.50% accuracy).

## 🎯 Why Instructor?

The original implementation uses manual JSON parsing which is error-prone. **Instructor** provides:

- ✅ **Structured Output**: Automatic Pydantic model validation
- ✅ **Type Safety**: Guaranteed output schema
- ✅ **Better Error Handling**: Automatic retries and validation
- ✅ **Cleaner Code**: No manual JSON parsing needed

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements-instructor.txt
```

## 🚀 Quick Start

### Single Article Classification

```python
from agents.instructor_classification_agent import InstructorClassificationAgent

# Initialize agent
agent = InstructorClassificationAgent(
    model_path="../models/gemma-native-ads-v7_merged_16bit",
    use_instructor=True
)

# Classify
result = agent.classify(
    title="Promo Spesial BRI di HUT ke-128",
    content="Bank BRI memberikan diskon hingga Rp1.28 juta..."
)

print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Reasoning: {result.reasoning}")
```

### Batch Classification

```python
articles = [
    {"title": "...", "content": "..."},
    {"title": "...", "content": "..."}
]

results = agent.classify_batch(articles, show_progress=True)
```

## 📚 Examples

Run the comprehensive examples:

```bash
python example_instructor.py
```

This will demonstrate:
1. Single article classification
2. Batch classification
3. Comparison with/without Instructor
4. Error handling

## 🏗️ Architecture

### Pydantic Model

```python
class ClassificationResult(BaseModel):
    label: Literal["native ads", "berita murni"]
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Max 200 characters
```

### Agent Features

- **Automatic truncation**: Content limited to 800 chars to avoid context overflow
- **Fallback mode**: Works with or without Instructor
- **Batch processing**: Efficient multi-article classification
- **Model info**: Get detailed model information

## 📊 Model Performance (Gemma v7)

- **Overall Accuracy**: 90.50% (181/200)
- **Berita Murni**: 85.58%
- **Native Ads**: 95.83%
- **BERTScore F1**: 0.662

## 🔧 Configuration

```python
agent = InstructorClassificationAgent(
    model_path="../models/gemma-native-ads-v7_merged_16bit",
    device="auto",              # 'auto', 'cuda', 'cpu'
    max_new_tokens=256,         # Max tokens to generate
    temperature=0.1,            # Lower = more deterministic
    use_instructor=True         # Enable structured output
)
```

## 🆚 Comparison with Original

| Feature | Original | Instructor |
|---------|----------|------------|
| Output Parsing | Manual JSON regex | Automatic Pydantic |
| Type Safety | ❌ None | ✅ Full validation |
| Error Handling | Basic try-catch | ✅ Built-in retries |
| Code Complexity | High | Low |
| Reliability | Medium | High |

## 📝 Notes

- Model must be located at the specified path
- Requires GPU for optimal performance (CPU works but slower)
- First run will be slower due to model loading
- Instructor mode is recommended for production use

## 🐛 Troubleshooting

**Model not found:**
```bash
# Make sure model exists
ls -la ../models/gemma-native-ads-v7_merged_16bit
```

**Out of memory:**
- Reduce `max_new_tokens`
- Use CPU instead of GPU
- Process articles one at a time instead of batch

**Slow inference:**
- Use GPU if available
- Reduce `max_new_tokens`
- Disable progress bar in batch mode

## 📖 Related Files

- `agents/instructor_classification_agent.py` - Main agent implementation
- `example_instructor.py` - Usage examples
- `requirements-instructor.txt` - Dependencies
- `agents/classification_agent.py` - Original LangChain implementation (for comparison)
