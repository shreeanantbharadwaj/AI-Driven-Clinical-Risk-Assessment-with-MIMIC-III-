# AI-Driven-Clinical-Risk-Assessment-with-MIMIC-III-


This project evaluates the performance of **Retrieval-Augmented Generation (RAG)** techniques integrated with **GPT-3.5**, comparing it to fine-tuned **ClinicalBERT** for predictive tasks in the clinical domain. By leveraging advanced AI pipelines and real-world datasets, the project demonstrates the potential of generalized language models to outperform domain-specific models in ICU risk assessment.

## Features

1. **Data Preprocessing**: Preprocessed clinical data using **spaCy** and **Hugging Face Transformers**.
2. **Model Development**:
   - Fine-tuned **ClinicalBERT** for sequence classification.
   - Implemented **RAG pipelines** with **Pinecone** vector search and **GPT-3.5**.
3. **Evaluation**:
   - Benchmarked models with **ROUGE** and custom **RAGAS metrics** for faithfulness, relevance, and recall.
4. **Use Case**: ICU risk assessment to predict high-risk patients using clinical text.

## Technologies and Tools

- **Data Preprocessing**: spaCy, Hugging Face Transformers, Pandas, NumPy
- **Embedding Generation**: OpenAI Ada model
- **Database**: Pinecone vector store
- **Model Development**: GPT-3.5, ClinicalBERT
- **Frameworks**: PyTorch
- **Evaluation**: ROUGE and custom metrics

## How to Run

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/project-name.git
   cd project-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add API keys in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

### Usage
1. Preprocess the data:
   ```bash
   python preprocess_data.py
   ```
2. Train ClinicalBERT:
   ```bash
   python train_clinicalbert.py
   ```
3. Run the RAG pipeline:
   ```bash
   python run_rag_pipeline.py
   ```
4. Evaluate models:
   ```bash
   python evaluate_models.py
   ```

## References

- [MIMIC-III Dataset](https://physionet.org/content/mimiciii/1.4/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## Acknowledgments

Special thanks to **Northeastern University** and all contributors for their support in completing this project.
