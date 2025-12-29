# Research Guide - Native Ads Detection

**Postdoc Research**: Pengembangan Agentic AI untuk Deteksi Native Ads pada Portal Berita Elektronik

## 📚 Latar Belakang

### Masalah Penelitian
- Native advertising semakin sulit dibedakan dari konten editorial
- Pembaca sering tidak menyadari sedang membaca iklan
- Diperlukan sistem otomatis untuk deteksi native ads
- Transparansi dan explainability penting untuk kepercayaan publik

### Gap Penelitian
- Metode tradisional (rule-based, ML klasik) kurang akurat
- Kurang explainability dalam keputusan klasifikasi
- Belum ada framework Agentic AI untuk native ads detection
- Dataset native ads Indonesia masih terbatas

## 🎯 Tujuan Penelitian

### Tujuan Utama
Mengembangkan sistem Agentic AI yang dapat mendeteksi native advertising pada portal berita elektronik dengan akurasi tinggi dan explainability yang baik.

### Tujuan Spesifik
1. Merancang arsitektur multi-agent untuk native ads detection
2. Mengimplementasikan LLM-based classifier dengan RAG
3. Mengembangkan mekanisme explainability
4. Membangun dataset native ads dari portal berita Indonesia
5. Evaluasi performa vs metode tradisional

## 🔬 Metodologi

### 1. Data Collection

```python
# Scraping artikel dari portal berita
portals = [
    "detik.com",
    "kompas.com", 
    "tribunnews.com",
    "cnnindonesia.com",
    "tempo.co"
]

# Target: 1000+ artikel (500 editorial, 500 native ads)
```

### 2. Feature Engineering

**Features untuk Native Ads Detection:**
- Promotional language ratio
- Brand mention frequency
- Call-to-action presence
- Disclosure statement detection
- Sentiment analysis
- Writing style analysis
- URL structure analysis

### 3. Agentic AI Architecture

**Components:**
1. **Web Agent**: Scraping otomatis
2. **Preprocessing Agent**: Feature extraction
3. **Retriever Agent**: RAG untuk context
4. **LLM Classifier**: GPT-4 classification
5. **Explanation Agent**: Interpretability
6. **Feedback Agent**: Continuous learning

### 4. Evaluation Metrics

```python
metrics = {
    'accuracy': 'Overall classification accuracy',
    'precision': 'Precision untuk native ads class',
    'recall': 'Recall untuk native ads class',
    'f1_score': 'F1-score',
    'explainability_score': 'Human evaluation of explanations',
    'confidence_calibration': 'Reliability of confidence scores'
}
```

## 📊 Eksperimen

### Experiment 1: Baseline Comparison

**Metode yang dibandingkan:**
- Rule-based classifier
- Traditional ML (SVM, Random Forest)
- Deep Learning (BERT, RoBERTa)
- **Agentic AI (proposed)**

### Experiment 2: Ablation Study

Test kontribusi setiap komponen:
- Without Retriever Agent
- Without Explanation Agent
- Without Feedback Agent
- Full system

### Experiment 3: Explainability Analysis

Evaluasi kualitas penjelasan:
- Human evaluation (expert journalists)
- Consistency analysis
- Factor importance ranking

## 📈 Expected Results

### Hipotesis
1. Agentic AI akan outperform metode tradisional
2. RAG approach meningkatkan akurasi
3. Explainability membantu trust dan adoption
4. Continuous learning meningkatkan performa over time

### Target Metrics
- Accuracy: > 90%
- Precision (Native Ads): > 85%
- Recall (Native Ads): > 85%
- F1-Score: > 85%
- Explainability Score: > 4.0/5.0

## 📝 Publikasi Strategy

### Target Journals (Q1/Q2)
1. **AI & NLP:**
   - ACM Transactions on Intelligent Systems and Technology
   - IEEE Transactions on Neural Networks and Learning Systems
   - Artificial Intelligence Review

2. **Digital Marketing & Media:**
   - Journal of Advertising Research
   - International Journal of Advertising
   - Digital Journalism

3. **Interdisciplinary:**
   - Expert Systems with Applications
   - Information Processing & Management

### Target Conferences
- AAAI (AI)
- ACL (NLP)
- WWW (Web)
- ICML (Machine Learning)
- WSDM (Web Search & Data Mining)

## 🛠️ Implementation Steps

### Phase 1: System Development (Bulan 1-2)
- [x] Setup Agentic AI architecture
- [x] Implement all agents
- [ ] Testing & debugging
- [ ] Optimization

### Phase 2: Data Collection (Bulan 2-3)
- [ ] Scraping 1000+ artikel
- [ ] Manual labeling (expert annotation)
- [ ] Inter-annotator agreement check
- [ ] Dataset validation

### Phase 3: Experiments (Bulan 3-4)
- [ ] Baseline implementation
- [ ] Agentic AI training
- [ ] Comparative experiments
- [ ] Ablation studies

### Phase 4: Evaluation (Bulan 4-5)
- [ ] Quantitative evaluation
- [ ] Explainability analysis
- [ ] Human evaluation
- [ ] Statistical significance testing

### Phase 5: Writing & Publication (Bulan 5-6)
- [ ] Draft paper
- [ ] Revisions
- [ ] Submission
- [ ] Response to reviewers

## 📚 Literature Review

### Key Papers

**Native Advertising Detection:**
1. Wojdynski & Evans (2016) - Going Native
2. Amazeen & Muddiman (2018) - Saving Media or Trading on Trust?

**Agentic AI:**
3. Xi et al. (2023) - The Rise and Potential of LLM-based Agents
4. Wang et al. (2024) - A Survey on Large Language Model based Agents

**RAG & LLM:**
5. Lewis et al. (2020) - Retrieval-Augmented Generation
6. Gao et al. (2023) - Retrieval-Augmented LLMs

**Explainable AI:**
7. Arrieta et al. (2020) - Explainable AI: A Review
8. Ribeiro et al. (2016) - "Why Should I Trust You?"

## 🎓 Contribution

### Theoretical Contributions
1. Framework Agentic AI untuk native ads detection
2. Multi-agent architecture dengan explainability
3. RAG approach untuk context-aware classification

### Practical Contributions
1. Working system untuk portal berita Indonesia
2. Dataset native ads Indonesia (akan di-publish)
3. Open-source implementation
4. Guidelines untuk media transparency

## 📧 Collaboration Opportunities

- **Journalism Schools**: Expert annotation, validation
- **News Portals**: Real-world testing, deployment
- **Advertising Industry**: Industry perspective
- **Regulatory Bodies**: Policy implications

## 🔗 Resources

- **Code Repository**: [GitHub URL]
- **Dataset**: [Dataset URL when published]
- **Demo**: [Demo URL]
- **Documentation**: This repository

---

**Last Updated**: 2024
**Researcher**: [Your Name]
**Institution**: [Your University]
