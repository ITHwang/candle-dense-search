## Info

---

- Article: [Streamlining Serverless ML Inference: Unleashing Candle Framework’s Power in Rust](https://towardsdatascience.com/streamlining-serverless-ml-inference-unleashing-candle-frameworks-power-in-rust-c6775d558545)
- GitHub: [candle_demo_1-1](https://github.com/a-agmon/candle_demo_1-1)
- Dataset: [arXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts/)

## Get Started

---

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts/)
2. Unzip the dataset and move `arxiv_data.csv` to `data/`
3. Cargo build

   ```bash
   cd dense_search/models_hf
   cargo build

   cd dense_search/embedding_generator
   cargo build

   cd dense_search/inf_server
   cargo build
   ```

4. Data Embedding

   ```bash
   cd dense_search/embedding_generator
   cargo run ../data/arxiv_data.csv
   ```

   > Warning: The number of rows in the dataset is over 50,000, which makes the embedding generation process slow. You can reduce the number of rows in the dataset to speed up the process.

5. Start the Inference Server

   ```bash
   cd dense_search/inf_server
   cargo run
   ```

6. Send a POST request to `http://localhost:3030/similar`

   ```bash
   curl -X POST http://localhost:3000/similar \
   -H "Content-Type: application/json" \
   -d '{
      "text": "deep learning survey",
      "num_results": 5
   }'
   ```

   ```text
   Output:

   {"text":["Item: The Principles of Deep Learning Theory (index: 1538 score: 0.9104092)","Item: Deep Reinforcement Learning for Autonomous Driving: A Survey (index: 2282 score: 0.9061589)","Item: Deep Learning for Person Re-identification: A Survey and Outlook (index: 2317 score: 0.9051876)","Item: Image Segmentation Using Deep Learning: A Survey (index: 302 score: 0.905109)","Item: Hyperbolic Deep Neural Networks: A Survey (index: 2229 score: 0.90289414)"]}
   ```
