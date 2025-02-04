use candle::{safetensors, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// NOTE: max length: 128
// Hidden vector size: 384
pub struct BertInferenceModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    embeddings: Tensor,
}

impl BertInferenceModel {
    pub fn load(
        model_name: &str,
        revision: &str,
        embeddings_filename: &str,
        embeddings_key: &str,
    ) -> anyhow::Result<Self> {
        let device = Device::Cpu;

        // Load the embeddings from a file
        let embeddings = match embeddings_filename.is_empty() {
            true => {
                println!("No file name provided. Embeddings return empty tensor.");
                Tensor::new(&[0.0], &device)?
            }
            false => {
                let tensor_file = safetensors::load(embeddings_filename, &device)
                    .expect("Error loading embeddings file");
                tensor_file
                    .get(embeddings_key)
                    .expect("Error getting embeddings key")
                    .clone()
            }
        };
        println!("Loaded embedding shape: {:?}", embeddings.shape());

        // Start loading the model from the hub
        let repo = Repo::with_revision(model_name.parse()?, RepoType::Model, revision.parse()?);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = api.get("model.safetensors")?;

        // load the model config
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;

        // load the tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        // load the model
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            embeddings,
        })
    }

    pub fn infer_sentence_embedding(&self, sentence: &str) -> anyhow::Result<Tensor> {
        let tokens = self
            .tokenizer
            .encode(sentence, true)
            .map_err(anyhow::Error::msg)?;

        let token_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        // WARN: Are they attention masks? If so, we need to create a tensor of 1s and 0s
        let token_type_ids = token_ids.zeros_like()?;

        let start = std::time::Instant::now();
        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        println!("Time taken for inference: {:?}", start.elapsed());
        println!("Embeddings: {:?}", embeddings);

        let embeddings = Self::apply_max_pooling(&embeddings)?;
        println!("Embeddings after max pooling: {:?}", embeddings);

        let embeddings = Self::l2_normalize(&embeddings)?;

        Ok(embeddings)
    }

    pub fn create_embeddings(&self, sentences: Vec<String>) -> anyhow::Result<Tensor> {
        println!("create_embeddings: sentences.len(): {}", sentences.len());

        let tokens = self
            .tokenizer
            .encode_batch(sentences, true)
            .map_err(anyhow::Error::msg)?;

        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        // WARN: Are they attention masks? If so, we need to create a tensor of 1s and 0s
        let token_type_ids = token_ids.zeros_like()?;

        println!("token_ids(input) shape: {:?}", token_ids.shape());

        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        let embeddings = Self::apply_max_pooling(&embeddings)?;
        let embeddings = Self::l2_normalize(&embeddings)?;

        println!(
            "create_embeddings completed - shape: {:?}",
            embeddings.shape()
        );

        Ok(embeddings)
    }

    pub fn score_vector_similarity(
        &self,
        vector: Tensor,
        top_k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let vec_len = self.embeddings.dim(0)?;
        let mut scores = vec![(0, 0.0); vec_len];

        for (embedding_index, score_tuple) in scores.iter_mut().enumerate() {
            let cur_vec = self.embeddings.get(embedding_index)?.unsqueeze(0)?;
            // NOTE: cur_vec and (query) vector are already normalized
            let cosine_similarity = (&cur_vec * &vector)?.sum_all()?.to_scalar::<f32>()?;
            *score_tuple = (embedding_index, cosine_similarity);
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);

        Ok(scores)
    }

    pub fn apply_max_pooling(embeddings: &Tensor) -> anyhow::Result<Tensor> {
        Ok(embeddings.max(1)?)
    }

    pub fn apply_mean_pooling(embeddings: &Tensor) -> anyhow::Result<Tensor> {
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        // TODO: Check if this is correct
        // The number of tokens is different sentence by sentence
        // Wondering if all the hidden vectors are valid
        // If there are zero-padding tokens, their hidden vectors should be ignored.
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;

        Ok(embeddings)
    }

    pub fn l2_normalize(embeddings: &Tensor) -> anyhow::Result<Tensor> {
        Ok(embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
}
